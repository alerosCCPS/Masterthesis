import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional
from utils import check_path
import sys


Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(Script_Root, ".."))
class DataSplitor:

    def __init__(self):
        self.raw_data_root = os.path.join(Script_Root, "ros2_ws", "src", "model_iden", "model_iden","DATA")
        self.data_list = os.listdir(self.raw_data_root)
        print("detect data: ", self.data_list)
        self.save_root = os.path.join(Script_Root, "DATA")
        check_path((self.save_root))
        self.sample_rate = 1000  # Hz
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.init_velocity: Optional[float] = 0.0
        self.target_velocity: Optional[float] = 0.0

    def load_data(self, path):
        df = pd.read_csv(path)
        self.y = np.array(df['velocity'].tolist())
        self.x = np.linspace(0, len(self.y), len(self.y))
        self.init_velocity = df["init_velocity"][0]
        self.target_velocity = df["target_velocity"][0]

    def plot_raw(self, name):
        plt.plot(self.x, self.y)
        plt.xlabel("step")
        plt.ylabel("velocity (m/s)")
        plt.title(f"raw data of {name}")
        plt.show()

    def show_raw_data(self):
        print("showing raw data...")
        for name in self.data_list:
            self.load_data(os.path.join(self.raw_data_root, name))
            self.plot_raw(name=name.split('.')[0])

    def split_data(self, case_name, start=0, end=-1):
        path = os.path.join(self.raw_data_root, f'{case_name}.csv')
        self.load_data(path)
        sliced_data = self.y[start: end+1]
        time_horizon = len(sliced_data) / self.sample_rate
        time_steps = np.linspace(0, 1, len(sliced_data)) * time_horizon

        plt.plot(time_steps, sliced_data)
        plt.xlabel("time step (s)")
        plt.ylabel("velocity (m/s)")
        plt.title(f"splited data from {case_name}")
        plt.savefig(os.path.join(self.save_root, f"splited_{case_name}.png"))
        plt.show()

        data = {
            "velocity": sliced_data,
            "time_steps": time_steps,
            "K": self.target_velocity - self.init_velocity
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_root, f"splited_{case_name}.csv"), index=False)

if __name__ == "__main__":

    ds = DataSplitor()
    # ds.show_raw_data()
    ds.split_data('velocity_02_06', 170, 300)
    ds.split_data('velocity_06_02', 179, 300)
    ds.split_data('velocity_idle_02_06', 156, 300)
    ds.split_data('velocity_idle_06_02', 183, 300)

