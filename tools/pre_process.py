import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path


# =====================================================
# 1. TE Preprocessor
# =====================================================
class TEPreprocessor:
    """
    TE data preprocessor:
    - StandardScaler (fit on TRAIN data only)
    - Sliding window
    - Zero padding
    """

    def __init__(self, window_size=100, step=10, padding_value=0.0):
        self.window_size = window_size
        self.step = step
        self.padding_value = padding_value
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, data_list):
        """
        Fit scaler using TRAIN data only.

        Args:
            data_list: list of np.ndarray, each shape (T, N)
        """
        all_data = np.vstack(data_list)
        self.scaler.fit(all_data)
        self.fitted = True

    def transform(self, data, label):
        """
        Generate sliding-window samples for one fault.

        Args:
            data : np.ndarray, shape (T, N)
            label: int

        Returns:
            X: torch.Tensor, shape (K, L, N)
            y: torch.Tensor, shape (K,)
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")

        data = self.scaler.transform(data)
        T, N = data.shape

        windows, labels = [], []

        for start in range(0, T, self.step):
            end = start + self.window_size
            window = data[start:end]

            # Zero padding (for last window)
            if window.shape[0] < self.window_size:
                pad_len = self.window_size - window.shape[0]
                pad = np.full((pad_len, N), self.padding_value)
                window = np.vstack([window, pad])

            windows.append(window)
            labels.append(label)

        X = torch.tensor(np.stack(windows), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        return X, y


# =====================================================
# 2. Raw data split (per fault, by time)
# =====================================================
def split_raw_per_fault(data_list, train_ratio=0.7):
    """
    Split raw TE data by time for each fault.

    Args:
        data_list: list of np.ndarray, each shape (T, N)

    Returns:
        train_list, test_list
    """
    train_list, test_list = [], []

    for data in data_list:
        T = data.shape[0]
        n_train = int(T * train_ratio)

        train_list.append(data[:n_train])
        test_list.append(data[n_train:])

    return train_list, test_list


# =====================================================
# 3. Main Process
# =====================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # -------------------------------
    # Global settings
    # -------------------------------
    data_path = "/home/Research_work/24_LY/RL/Energy/Processed-Data/"
    save_path = "/home/Research_work/24_LY/RL/Energy/Load-Data/"
    train_ratio = 0.6

    window_size = 48
    step = 6

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Load raw TE data
    # -------------------------------
    load_data_path = Path(data_path)
    all_file_path = sorted([f for f in load_data_path.iterdir() if f.is_file()])
    # Select fault
    select_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    all_file_path = [all_file_path[i] for i in select_id]
    print("Length of select data-file",len(all_file_path))

    data_list = [np.load(p) for p in all_file_path]
    fault_labels = list(range(len(data_list)))  # 0 ~ 20

    print(f"Total faults: {len(data_list)}")

    # -------------------------------
    # 1) Split raw data FIRST
    # -------------------------------
    train_raw, test_raw = split_raw_per_fault(
        data_list, train_ratio=train_ratio
    )

    # -------------------------------
    # 2) Fit scaler on TRAIN only
    # -------------------------------
    print("Fitting scaler using TRAIN data only...")
    preprocessor = TEPreprocessor(
        window_size=window_size,
        step=step
    )
    preprocessor.fit(train_raw)

    # -------------------------------
    # 3) Sliding window generation
    # -------------------------------
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print("Generating sliding windows...")
    for label, (tr, te) in enumerate(zip(train_raw, test_raw)):
        X_tr, y_tr = preprocessor.transform(tr, label)
        X_te, y_te = preprocessor.transform(te, label)

        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_test_list.append(X_te)
        y_test_list.append(y_te)

    # -------------------------------
    # 4) Concatenate all faults
    # -------------------------------
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)

    # -------------------------------
    # Summary
    # -------------------------------
    print("\nAfter preprocessing:")
    print("Train X:", X_train.shape)
    print("Test  X:", X_test.shape)
    print("Train labels:", torch.unique(y_train))
    print("Test  labels:", torch.unique(y_test))

    # -------------------------------
    # Save (NumPy format)
    # -------------------------------
    np.save(save_path + "X_train.npy", X_train.numpy())
    np.save(save_path + "y_train.npy", y_train.numpy())
    np.save(save_path + "X_test.npy", X_test.numpy())
    np.save(save_path + "y_test.npy", y_test.numpy())

    print("\nSaved preprocessed data to:", save_path)
