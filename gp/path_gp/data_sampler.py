import os
import numpy as np
import pandas as pd

Script_Root = os.path.abspath(os.path.dirname(__file__))

class Sampler:

    def __init__(self,sample_rate=0.1):
        self.data_root = os.path.join(Script_Root, "DATA")
        self.data_list = [os.path.join(self.data_root, f) for f in ["dataset_mpc.csv", 'dataset_reverse.csv']]
        # for f in os.listdir(self.data_root):
        #     if f.endswith(".csv"):
        #         self.data_list.append(os.path.join(self.data_root, f))
        print(f"found data files: ")
        print(self.data_list)
        self.sample_rate = sample_rate
        self.train_data = []
        self.test_data = []
    def sampling(self):
        for f in self.data_list:
            self.split_data(f)
        self.save_data()

    def split_data(self, file):
        df = pd.read_csv(file)
        kappa = df['curvature'].values.reshape(-1,1)
        LF = df['look_fw_curvature'].values.reshape(-1,1)
        n = df['n'].values.reshape(-1,1)
        alpha = df['alpha'].values.reshape(-1,1)
        # v = df['v'].values.reshape(-1,1)
        # v_comm = df['v_comm'].values.reshape(-1,1)
        delta = df['delta'].values.reshape(-1,1)
        rec_diff = df['rec_diff'].values.reshape(-1,1)

        data = np.hstack((kappa,LF, n, alpha, delta, rec_diff))
        num = data.shape[0]
        sample_step = int(self.sample_rate**-1)
        idx = [0] +[i for i in range(1, num-1, sample_step)]+ [num-1]
        train_data = data[idx]
        test_data = np.delete(data, idx, axis=0)
        self.train_data.append(train_data)
        self.test_data.append(test_data)
    def save_data(self):
        train_data = np.vstack(self.train_data)
        test_data = np.vstack(self.test_data)
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        head = ['curvature', 'look_fw_curvature', 'n', 'alpha', 'delta', 'rec_diff']
        pd.DataFrame(train_data, columns=head).to_csv(os.path.join(self.data_root, 'train_data.csv'), index=False)
        pd.DataFrame(test_data, columns=head).to_csv(os.path.join(self.data_root, 'test_data.csv'), index=False)
        print(f"saving training data {train_data.shape[0]}")
        print(f"saving testing  data {test_data.shape[0]}")


if __name__ == '__main__':
    # case_name = 'test_traj_3D'
    # case_name = 'test_traj_2D'
    sampler = Sampler()
    sampler.sampling()