import numpy as np
import os
import pandas as pd

Script_Root = os.path.abspath(os.path.dirname(__file__))
class ReSampler:

    def __init__(self):

        self.data_root = Script_Root
        self.gp_list = ["2D", "2D_LF", "3D", "3D_LF"]
        self.traj_list = ["test_traj"]
        self.sample_step = [-20, -10, -4, -3, 2,3,4,5]
        self.data_postfix = {2:"50", 3:"66", 4:"75", 5:"80",
                             -3:"33", -4: "25", -10:"10", -20:"5"}

    def load_data(self, path):
        df = pd.read_csv(path)
        kappa = df['curvature'].values.reshape(-1, 1)
        LF = df['look_fw_curvature'].values.reshape(-1, 1)
        n = df['n'].values.reshape(-1, 1)
        alpha = df['alpha'].values.reshape(-1, 1)
        delta = df['delta'].values.reshape(-1, 1)
        rec_diff = df['rec_diff'].values.reshape(-1, 1)
        data = np.hstack((kappa, LF, n, alpha, delta, rec_diff))
        return data

    def resample(self, root_path):
        data = self.load_data(os.path.join(root_path, "train_data.csv"))
        num = data.shape[0]
        for step in self.sample_step:
            data_copy = data.copy()
            idx = [0] + [i for i in range(1, num - 1, abs(step))] + [num - 1]
            rest_train_data = data[idx]
            resampled_data = np.delete(data_copy, idx, axis=0)
            head = ['curvature', 'look_fw_curvature', 'n', 'alpha', 'delta', 'rec_diff']
            if step>0:
                pd.DataFrame(resampled_data, columns=head).to_csv(os.path.join(root_path, f'train_data_{self.data_postfix[step]}.csv'), index=False)
                sample_size = resampled_data.shape[0]
            if step < 0:
                pd.DataFrame(rest_train_data, columns=head).to_csv(os.path.join(root_path, f'train_data_{self.data_postfix[step]}.csv'), index=False)
                sample_size = rest_train_data.shape[0]
            print(f"resampling data {sample_size}")
            print(f"saving data at {root_path}")
if __name__ == "__main__":
    sampler = ReSampler()
    sampler.resample(os.path.join(Script_Root, "DATA"))