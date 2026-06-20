from Process.cstr_load import *
from Process.Monitor import *
from Process.tools import *
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def ensure_dir(path):
    if os.path.exists(path):
        return True, None

    try:
        os.makedirs(path, exist_ok=True)
        return True, None
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    np.random.seed(42)
    # --->Select Method
    method = "KoopJointSFA"
    # ---> Temp Path
    state_path = ".../"
    # ---> Construct Data
    train_path, test_path = construct_cstr_data_path()
    all_train_data = build_data(train_path)
    all_test_data = build_data(test_path)
    FAR, FDR = [], []
    # ---> Select All Fault
    all_fault = np.array(range(len(list(Define_fault.keys())) - 1)) # Remove normal state
    # Standardization
    global_scaler = StandardScaler()
    train_data = all_train_data[0].T
    train_norm = global_scaler.fit_transform(train_data)
    for i in tqdm(all_fault):
        test_idx = i
        test_data = all_test_data[test_idx].T
        if method == "GLPQR":
            X_train, Y_train = split_cstr_for_glpqr(train_data, quality_index=2)
            X_test, Y_test = split_cstr_for_glpqr(test_data, quality_index=2)
            train_norm = global_scaler.fit_transform(X_train)
            test_norm = global_scaler.transform(X_test)
        elif method == "JTSVA":
            test_norm = global_scaler.transform(test_data)
        else:
            test_norm = global_scaler.transform(test_data)
        # --->Check Save path
        figure_folder = ".../" + method
        created, msg = ensure_dir(figure_folder)
        if not created:
            print(msg)
        # --->Fit & Monitor
        if method == "KPCA":
            limit, test, alarm, state = KPCAMonitor(train_norm, test_norm, alpha=0.99, n_components=30, kernel="rbf", gamma=0.001, return_state=True)

        elif method == "SFA":
            limit, test, alarm, state = SFAMonitor(train_norm, test_norm, alpha=0.99, n_components=20, return_state=True)

        elif method == "KSFA":
            limit, test, alarm, state = KSFAMonitor(train_norm, test_norm, alpha=0.99, n_components=40, kernel="rbf", gamma=0.1, return_state=True)

        elif method == "KoopJointSFA":
            limit, test, alarm, state = KoopJointSFAMonitor(
                train_norm, test_norm, return_state=True, gamma = 1e-5, joint_alpha = 1.0, 
                lag_step= 3, n_centers=5, n_components=15, n_lags=3, use_combined_alarm=True)

        elif method == "GLPQR":
            limit, test, alarm, state = GLPQRMonitor(train_norm, Y_train, test_norm, alpha=0.99, sigma=3500, lam=0.15, alpha_glpqr=0.5, beta=0.001, n_components=45, return_state=True)

        elif method == "JTSVA":
            limit, test, alarm, state = JTSVAMonitor(train_norm, test_norm, alpha=0.99, d=3, k=13, return_state = True)

        elif method == "DiCCA":
            limit, test, alarm, state = DiCCAMonitor(train_norm, test_norm, alpha=0.99, d=3, k=13, return_state=True)
        else:
            raise TypeError("No such method")
        
        # --->Calculate FAR || FDR
        FAR_item, FDR_item = monitoring_metrics(alarm, fault_start=180)
        FAR.append(FAR_item)
        FDR.append(FDR_item)
        
    print(f"=================Method:{method}=================")
    # --->Print FAR || FDR
    printing_metrics(FAR, FDR)
    print(f"=================Method:{method}=================")



