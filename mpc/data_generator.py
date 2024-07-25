import os
import pandas as pd
import numpy as np
import sys
from scipy.interpolate import splev, splrep
from utils import check_path
from hamster_dynamic import get_hamster_model
from mpc_controller import MPC

Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(Script_Root,".."))
n_limit = 0.1
class DataGen:

    def __init__(self, case_name="test_traj"):
        self.case_name = case_name
        # self.k = 0.2359
        self.k = 0.25
        self.data_root = os.path.join(Script_Root, "DATA", case_name)
        df = pd.read_csv(os.path.join(self.data_root, "path.csv"))
        def f(s):
            tck = splrep(df["s"].values, df["curvature"].values)
            return splev(s,tck=tck)
        self.kappa_of_s = f

        self.save_root = os.path.join(self.data_root, "synthetic")
        check_path(self.save_root)
        _, self.constrains = get_hamster_model(case_name)
        # self.sample_dis = 1.4
        # self.resolution_n = 0.04
        # self.resolution_alpha = 0.05

        self.sample_dis = 10
        self.resolution_n = 0.005
        self.resolution_alpha = 0.01

        self.position = 0
        self.mpc = MPC(case_name)
        self.data = []

    def generator(self):
        sample_steps = max(1, int(self.constrains.s_limit / self.sample_dis))

        for i in range(sample_steps):
            self.position = i*self.sample_dis
            if self.position >= self.constrains.s_limit:
                self.position -= self.constrains.s_limit
            local_kappa = self.kappa_of_s(self.position).item()
            n_upper = n_limit if n_limit < 1/abs(local_kappa) else 1/abs(local_kappa)
            n_lower = -n_limit if n_upper > -n_limit else n_upper-2*self.resolution_n
            alpha_upper = self.constrains.alpha_limit
            alpha_lower = -self.constrains.alpha_limit
            self.sampling(self.position,local_kappa, n_upper, n_lower, alpha_upper, alpha_lower)
        self.save_data()

    def sampling(self, s, local_kappa, n_upper, n_lower, alpha_upper, alpha_lower):
        iter_n = int((n_upper - n_lower)/self.resolution_n)+1
        iter_alpha = int((alpha_upper- alpha_lower)/self.resolution_alpha)+1
        print(f"iter_n = {iter_n}, iter_alpha={iter_alpha}")

        for i in range(iter_n):
            n = n_lower + i*self.resolution_n
            for j in range(iter_alpha):
                alpha = alpha_lower + j*self.resolution_alpha
                x = [s, n, alpha, 0.6]
                _, u, _ = self.mpc.predict(x=x)
                print([local_kappa, s, n, alpha, 0.6]+list(u))
                self.data.append([local_kappa, s, n, alpha, 0.6]+list(u))

    def save_data(self):
        heads = ["curvature", 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'rec_diff']
        data = np.array(self.data)
        rec_diff = data[:,-1] - self.k*data[:,0]
        data = np.hstack((data, rec_diff.reshape(-1,1)))
        df = pd.DataFrame(data,columns=heads)
        case = self.case_name.strip().split("_")[-1]
        df.to_csv(os.path.join(self.save_root, f"dataset_2D_{case}.csv"), index=False)
        print(f"collected data with shape: {data.shape}")
        print(f"saving data at {self.save_root}")

if __name__ == '__main__':
    case_name = 'test_traj_normal'
    # case_name = 'test_traj_reverse'
    gen = DataGen(case_name)
    gen.generator()