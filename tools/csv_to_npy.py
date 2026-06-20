import pandas as pd
import os
import numpy as np

class Preprocessor:
    def __init__(self):
        self.data_dir = "/home/Research_work/24_LY/RL/Energy/Download"
        self.feature_cols = None
        self.Max_length = 10000
        self.file_name = ['TEP_Faulty_Training.csv','TEP_Faulty_Testing.csv', 'TEP_FaultFree_Training.csv', 'TEP_FaultFree_Testing.csv']

    def load_csv_chunks(self, filename, chunksize = 10000)->pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(os.path.join(self.data_dir, filename), chunksize=chunksize):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def load_data(self):
        faulty_train = self.load_csv_chunks(self.file_name[0], chunksize=self.Max_length)
        faulty_test = self.load_csv_chunks(self.file_name[1], chunksize=self.Max_length)
        normal_train = self.load_csv_chunks(self.file_name[2], chunksize=self.Max_length)
        normal_test = self.load_csv_chunks(self.file_name[3], chunksize=self.Max_length)
        print("Load Over")
        # 重命名
        faulty_train_new = faulty_train.rename(columns={"faultNumber":"FaultType", "sample":"Num_sample", "simulationRun":"Run_id"})
        faulty_test_new = faulty_test.rename(columns={"faultNumber":"FaultType",  "sample":"Num_sample", "simulationRun":"Run_id"})
        normal_train_new = normal_train.rename(columns={"faultNumber":"FaultType", "sample":"Num_sample", "simulationRun":"Run_id"})
        normal_test_new = normal_test.rename(columns={"faultNumber":"FaultType", "sample":"Num_sample", "simulationRun":"Run_id"})
        print("Rename Over")
        # 合并数据
        train_data = pd.concat([normal_train_new, faulty_train_new], ignore_index=True)
        test_data = pd.concat([normal_test_new, faulty_test_new], ignore_index=True)
        # 转为npy格式
        train_save_path = os.path.join("/home/Research_work/24_LY/RL/Energy/mat", "TEP_Fault_Train.pkl")
        test_save_path = os.path.join("/home/Research_work/24_LY/RL/Energy/mat", "TEP_Fault_Test.pkl")
        train_data.to_pickle(train_save_path)
        test_data.to_pickle(test_save_path)

# p = Preprocessor()
# p.load_data()
train_data = pd.read_pickle(os.path.join("/home/Research_work/24_LY/RL/Energy/mat", "TEP_Fault_Train.pkl"))
# test_data = pd.read_pickle(os.path.join("/home/Research_work/24_LY/RL/Energy/mat", "TEP_Fault_Test.pkl"))
train_data = pd.DataFrame(train_data)
print(train_data.keys())