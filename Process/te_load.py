import numpy as np
import os
import matplotlib.pyplot as plt
global_root_folder = ".../"
def construct_data_path(root_folder):
    # Normal: d00
    # Abnormal: d01~d21
    normal = [os.path.join(root_folder, "d00_te.dat")]
    abnormal = []
    for i in range(1, 22):
        if i < 10:
            name = f"d0{i}_te.dat"
        else:
            name = f"d{i}_te.dat"
        mypath = os.path.join(root_folder, name)
        abnormal.append(mypath)
    return np.array(normal), np.array(abnormal)

def read_data(path):
    data =  np.loadtxt(path, dtype=np.float32)
    N, T = data.shape
    if N > T: # transpose
        data = data.T
    return data

def build_data(data_path:list):
    all_data = []
    for path_item in data_path:
        data = read_data(path_item)
        all_data.append(data)
    all_data = np.stack(all_data, axis=0)
    return all_data

def build_d00(root_folder):
    normal = [os.path.join(root_folder, "d00.dat")]
    d00 = build_data(normal)
    return d00

if __name__ == "__main__":
    train_path, test_path = construct_data_path(global_root_folder)
    all_train_data = build_data(train_path)
    all_test_data = build_data(test_path)
    # print(all_train_data.shape)
    # print(all_test_data.shape)
    # print(build_d00(global_root_folder).shape)