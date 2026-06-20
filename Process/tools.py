import numpy as np
def split_te_for_glpqr(data):
    """
    data shape: (n_samples, 52)
    column order assumed:
        XMEAS(1)-XMEAS(41), XMV(1)-XMV(11)
    """

    # XMEAS(1)-XMEAS(22)
    xmeas_1_22 = data[:, 0:22]

    # XMV(1)-XMV(11)
    xmv_1_11 = data[:, 41:52]

    # process variables X: 22 + 11 = 33 variables
    X = np.hstack([xmeas_1_22, xmv_1_11])

    # quality variable y: XMEAS(35)
    # XMEAS(35) -> 34
    Y = data[:, 34].reshape(-1, 1)

    return X, Y

def split_cstr_for_glpqr(data, quality_index=0):
    Y = data[:, quality_index].reshape(-1, 1)
    X = np.delete(data, quality_index, axis=1)
    return X, Y

def select_te_33_variables(data, is_select = True):
    if is_select:
        xmeas_1_22 = data[:, 0:22]
        xmv_1_11 = data[:, 41:52]
        return np.hstack([xmeas_1_22, xmv_1_11])
    else:
        return data
    
from sklearn.preprocessing import StandardScaler
def standardize(X_train, X_test, use_transform = True):
    scaler = StandardScaler()
    if use_transform:
        train_norm = scaler.fit_transform(X_train)
        test_norm = scaler.transform(X_test)
    else:
        train_norm, test_norm = X_train, X_test
    return train_norm, test_norm


def detection_delay(alarm, fault_start, consecutive=3):
    for i in range(fault_start, len(alarm) - consecutive + 1):
        if np.all(alarm[i:i + consecutive]):
            return i - fault_start
    return np.inf

def monitoring_metrics(alarm, fault_start = 160):
    normal_alarm = alarm[:fault_start]
    fault_alarm = alarm[fault_start:]

    FAR = np.mean(normal_alarm)
    FDR = np.mean(fault_alarm)

    return FAR, FDR

def printing_metrics(FAR, FDR):
    FAR = np.asarray(FAR)
    FDR = np.asarray(FDR)
    
    print(f"{'Fault':<8} {'FAR(%)':<12} {'FDR(%)':<12}")
    print("-" * 46)
    
    for i in range(len(FAR)):
        print(f"{i+1:<8} {FAR[i]:<12.2%} {FDR[i]:<12.2%}")
    
    print("-" * 46)
    print(f"{'Mean':<8} {np.mean(FAR):<12.2%} {np.mean(FDR):<12.2%}")