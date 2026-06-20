from Process.te_load import *
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
    train_path, test_path = construct_data_path(global_root_folder)
    all_train_data = build_data(train_path)
    all_test_data = build_data(test_path)
    d00 = build_d00(global_root_folder) # (1, 52, 500)
    # print(all_train_data.shape) # (1, 52, 960)
    # print(all_test_data.shape) # (21, 52, 960)
    FAR, FDR = [], []
    # ---> Select Data index
    selected_fault = np.array([1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21]) - 1
    all_fault = np.array(range(0, 21))
    # Select 33 vars & Standardization
    global_scaler = StandardScaler()
    train_data = all_train_data[0].T
    X_train = select_te_33_variables(train_data)
    train_norm = global_scaler.fit_transform(X_train)
    for i in tqdm(selected_fault):
        test_idx = i
        test_data = all_test_data[test_idx].T
        # print("train shape",train_data.shape) # (960, 52)
        # print("test shape",test_data.shape) # (960, 52)
        '''
        Note: For method GLPQR, the input includes the quality-related indicator
        '''
        if method == "GLPQR":
            X_train, Y_train = split_te_for_glpqr(train_data)
            X_test, Y_test = split_te_for_glpqr(test_data)
            test_norm = global_scaler.transform(X_test)
        elif method == "JTSVA":
            X_train = select_te_33_variables(train_data)
            X_test = select_te_33_variables(test_data)
            X_val = select_te_33_variables(d00[0].T)
            test_norm = global_scaler.transform(X_test)
            val_norm = global_scaler.transform(X_val)
        else:
            X_test = select_te_33_variables(test_data)
            test_norm = global_scaler.transform(X_test)
        # --->Check Save path
        figure_folder = ".../" + method
        created, msg = ensure_dir(figure_folder)
        if not created:
            print(msg)
        # --->Fit & Monitor
        if method == "KPCA":
            limit, test, alarm, state = KPCAMonitor(train_norm, test_norm, alpha=0.99, n_components=20, kernel="rbf", gamma=0.001, return_state=True)

        elif method == "SFA":
            limit, test, alarm, state = SFAMonitor(train_norm, test_norm, alpha=0.99, n_components=20, return_state=True)

        elif method == "KSFA":
            limit, test, alarm, state = KSFAMonitor(train_norm, test_norm, alpha=0.99, n_components=45, kernel="linear", gamma=0.1, return_state=True)

        elif method == "KoopJointSFA":
            limit, test, alarm, state = KoopJointSFAMonitor(train_norm, test_norm, return_state=True)

        elif method == "GLPQR":
            limit, test, alarm, state = GLPQRMonitor(train_norm, Y_train, test_norm, alpha=0.99, sigma=3500, lam=0.15, alpha_glpqr=0.5, beta=0.001, n_components=45, return_state=True)

        elif method == "JTSVA":
            limit, test, alarm, state = JTSVAMonitor(train_norm, test_norm, alpha=0.99, d=3, k=13, return_state = True)

        elif method == "DiCCA":
            limit, test, alarm, state = DiCCAMonitor(train_norm, test_norm, alpha=0.99, d=5, k=9, return_state=True)

        else:
            raise TypeError("No such method")
        
        # --->Calculate FAR || FDR
        FAR_item, FDR_item = monitoring_metrics(alarm)
        FAR.append(FAR_item)
        FDR.append(FDR_item)
    
    print(f"=================Method:{method}=================")
    # --->Print FAR || FDR
    printing_metrics(FAR, FDR)
    print(f"=================Method:{method}=================")



