# A general process for different data and transform them into same type like "npy", "pt", ....
import numpy as np
from pathlib import Path
import os
from typing import Literal, Union, List
def read_dat_data(file_path) -> np.ndarray:
    # read dat file
        with open(file_path, mode='r') as file:
            data = file.readlines()
        # str -> float
        mydata = []
        try:
            for line in data:
                # 去除空格
                data_line = line.strip().split()
                data_line = np.float32(data_line)
                mydata.append(data_line)
        except Exception as e:
            print(e)

        mydata = np.array(mydata, dtype=np.float32)
        return mydata

def save(save_path, data, myformat:Literal['npy', 'npz']):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if myformat == 'npy':
        np.save(str(save_path) + f'.{myformat}', data)
    elif myformat == 'npz':
        np.savez(str(save_path) + f'.{myformat}', data)
    else:
        raise ValueError(f"Unsupported format: {myformat}")

def read_file_name(file: Union[str, Path], extension:Literal['dat'] = 'dat') -> List[str]:
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {file}")
    if path.is_file():
        # Return the file name itself
        if path.name.lower().lstrip('.') != extension:
            return []
        else:
            return [path.name]
    elif path.is_dir():
        # Return all files in directory
        all_files = sorted([f.name for f in path.iterdir() if f.is_file()])
        return [f for f in all_files if f.lower().endswith(f'.{extension}')]
    else:
        raise ValueError(f"Path is neither a file nor directory: {file}")


data_path = '/home/Research_work/24_LY/RL/Energy/TE-Data/'
save_path = '/home/Research_work/24_LY/RL/Energy/Processed-Data/'
file_name_list = read_file_name(data_path)
for i, file_name in enumerate(file_name_list):
    file_path = os.path.join(data_path, file_name)
    data = read_dat_data(file_path)
    if 'te' not in file_name: # 去除48h数据
        file_name = Path(file_name)
        save_name = os.path.join(save_path, file_name.stem)
    save(save_name, data, myformat='npy')
    print(f"Save {i / len(file_name_list) * 100:.2f} %")