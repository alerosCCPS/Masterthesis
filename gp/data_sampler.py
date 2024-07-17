import os
import pandas
import numpy as np
import pandas as pd

Script_Root = os.path.abspath(os.path.dirname(__file__))

class Sampler:

    def __init__(self, case_name='test_traj'):
        self.data_root = os.path.join(Script_Root, "DATA", case_name)
        self.sample_rate = 0.2

    def split_data(self):
        df = pd.read_csv(os.path.join(self.data_root, 'sim_results.csv'))
        kappa, n, alpha, v, v_comm, delta = df['curvature'].values.reshape(-1,1),\
            df['n'].values.reshape(-1,1),\
            df['alpha'].values.reshape(-1,1),\
            df['v'].values.reshape(-1,1),\
            df['v_comm'].values.reshape(-1,1), \
            df['delta'].values.reshape(-1,1)
        head = ['curvature', 'n', 'alpha', 'v', 'v_comm', 'delta']
        data = np.hstack((kappa, n, alpha, v, v_comm, delta))
        num = data.shape[0]
        sample_step = int(self.sample_rate**-1)
        idx = [0] +[i for i in range(1, num-1, sample_step)]+ [num-1]
        train_data = data[idx]
        test_data = np.delete(data, idx, axis=0)

        pd.DataFrame(train_data, columns=head).to_csv(os.path.join(self.data_root, 'train_data.csv'), index=False)
        pd.DataFrame(test_data, columns=head).to_csv(os.path.join(self.data_root, 'test_data.csv'), index=False)
        print(f"splitting training data {train_data.shape[0]}")
        print(f"splitting testing  data {test_data.shape[0]}")


if __name__ == '__main__':

    sampler = Sampler()
    sampler.split_data()