import os
import numpy as np
import matplotlib.pyplot as plt
global_root_folder = ".../"
Define_fault = {
    0:"Normal.csv", # Normal
    1:"Fault1.csv", # Fault 1
    2:"Fault2.csv", # Fault 2
    3:"Fault3.csv", # Fault 3
    4:"Fault4.csv", # Fault 4
    5:"Fault5.csv", # Fault 5
    6:"Fault6.csv", # Fault 6
    7:"Fault7.csv", # Fault 7
    8:"Fault8.csv", # Fault 8
    9:"Fault9.csv", # Fault 9
    10:"Fault10.csv",# Fault 10
}

def construct_cstr_data_path(root_folder = global_root_folder, name_dict:dict = Define_fault):
    abnormal = []
    normal = []
    for key, value in zip(name_dict.keys(), name_dict.values()):
        if key == 0:
            # Normal
            normal_data_path = os.path.join(root_folder, value)
            normal.append(normal_data_path)
        else:
            # Abnormal
            abnormal_data_path = os.path.join(root_folder, value)
            abnormal.append(abnormal_data_path)
    return np.array(normal), np.array(abnormal)

def read_data(path, delimiter=',', skiprows=1, transpose_if_true=True):
    try:
        data = np.loadtxt(path, dtype=np.float32, delimiter=delimiter, skiprows=skiprows)
    except Exception as e:
        print(f"Fail to load: {e}")
        raise

    if transpose_if_true and data.shape[0] > data.shape[1]:
        data = data.T
    
    return data

def build_data(data_path:list):
    all_data = []
    for path_item in data_path:
        data = read_data(path_item)
        all_data.append(data)
    all_data = np.stack(all_data, axis=0)
    return all_data

if __name__ == "__main__":
    train_path, test_path = construct_cstr_data_path(global_root_folder, Define_fault)
    all_train_data = build_data(train_path)
    all_test_data = build_data(test_path)
    print(all_train_data.shape)
    print(all_test_data.shape)