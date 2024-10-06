import os
import pandas as pd
import numpy as np
import sys
Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(Script_Root,".."))
from scipy.interpolate import splev, splrep
from utils import check_path
from hamster_dynamic import get_hamster_model
# from mpc.mpc_controller import MPC
from mpc.adapted_mpc_controller import MPC
import casadi as ca
import math

deg2R = lambda x: math.pi*(x/180)
r2Deg = lambda x: 180*x/np.pi
n_limit = 0.2
alpha_limit = np.pi * 0.4
class DataGen:

    def __init__(self, case_name="test_traj"):
        self.case_name = case_name
        # self.k = 0.2359
        # self.k = 0.25

        self.targ_vel = 0.4
        self.max_steer = deg2R(25)
        self.L = 0.25
        self.steer_k1 = 2.2
        self.steer_k2 = 5
        self.length_rear = 0.125
        self.look_fw_dis = 0.15
        self.data_root = os.path.join(Script_Root, "DATA", case_name)
        df = pd.read_csv(os.path.join(self.data_root, "path.csv"))
        def f(s):
            tck = splrep(df["s"].values, df["curvature"].values)
            return splev(s,tck=tck)
        self.kappa_of_s = f

        self.save_root = os.path.join(self.data_root, "synthetic_adapted")
        check_path(self.save_root)
        _, self.constrains = get_hamster_model(case_name)
        self.sample_dis = 0.5
        self.resolution_n = 0.025
        self.resolution_alpha = 0.25

        # self.sample_dis = 10
        # self.resolution_n = 0.005
        # self.resolution_alpha = 0.01

        self.position = 0
        self.mpc = MPC(case_name)
        self.N = self.mpc.N
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
            # alpha_upper = self.constrains.alpha_limit
            # alpha_lower = -self.constrains.alpha_limit
            alpha_upper = alpha_limit
            alpha_lower = -alpha_limit
            self.sampling(self.position,local_kappa, n_upper, n_lower, alpha_upper, alpha_lower)
        self.save_data()

    def sampling(self, s, local_kappa, n_upper, n_lower, alpha_upper, alpha_lower):
        # iter_n = int((n_upper - n_lower)/self.resolution_n)+1
        # iter_alpha = int((alpha_upper- alpha_lower)/self.resolution_alpha)+1
        n_threshold = 0.03
        n_fine = 0.01
        n_sampled = [n_lower + i*self.resolution_n for i in range(1+int((-n_threshold - n_lower)/self.resolution_n))] + \
                    [-n_threshold + i*n_fine for i in range(int(2*n_threshold/n_fine))] + \
                    [n_threshold + i*self.resolution_n for i in range(1+int((n_upper - n_threshold)/self.resolution_n))]

        alpha_threshold = 0.4
        alpha_fine = 0.1
        alpha_sampled = [alpha_lower + i*self.resolution_alpha for i in range(1+int((-alpha_threshold - alpha_lower)/self.resolution_alpha))] + \
                        [-alpha_threshold + i*alpha_fine for i in range(int(2*alpha_threshold/alpha_fine))] + \
                        [alpha_threshold + i*self.resolution_alpha for i in range(1+int((alpha_upper-alpha_threshold)/self.resolution_alpha))]

        print(f"iter_n = {len(n_sampled)}, iter_alpha={len(alpha_sampled)}")
        look_fw = self.kappa_of_s(s + self.look_fw_dis).item()
        for n in n_sampled:
            for alpha in alpha_sampled:
        # for i in range(iter_n):
        #     n = n_lower + i*self.resolution_n
        #     for j in range(iter_alpha):
        #         alpha = alpha_lower + j*self.resolution_alpha
                x = [s, n, alpha, 0.3]
                x0x = ca.repmat([0,0,0,0], (self.N + 1))
                x0u = ca.repmat([self.targ_vel, self.L/2 * local_kappa], self.N)
                x0 = ca.vertcat(x0x, x0u)
                _, u, _ , _,_,_, status_flag= self.mpc.predict(x0=x0, x=x)
                if not status_flag:
                    print("skip infeasible results!")
                    print([local_kappa, look_fw, s, n, alpha, self.targ_vel] + list(u))
                    continue
                # print([local_kappa,look_fw, s, n, alpha, self.targ_vel] + list(u))
                self.data.append([local_kappa,look_fw, s, n, alpha, self.targ_vel]+list(u))

    def save_data(self):
        heads = ["curvature",'look_fw_curvature', 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'rec_diff']
        data = np.array(self.data)
        # rec_diff = data[:,-1] - self.k*data[:,0]

        # delta_ref = np.arctanh(np.tan(data[:, 0] *self.L)/(self.steer_k1 * self.L) )/ self.steer_k2
        delta_ref = np.arctanh(data[:, 0]/self.steer_k1) / self.steer_k2
        delta_ref[delta_ref>self.max_steer] = self.max_steer
        delta_ref[delta_ref<-self.max_steer] = -self.max_steer
        rec_diff = data[:, -1] - delta_ref
        data = np.hstack((data, rec_diff.reshape(-1,1)))
        df = pd.DataFrame(data,columns=heads)
        case = self.case_name.strip().split("_")[-1]
        df.to_csv(os.path.join(self.save_root, f"dataset_{case}.csv"), index=False)
        print(f"collected data with shape: {data.shape}")
        print(f"saving data at {self.save_root}")

if __name__ == '__main__':
    cases = ['test_traj_mpc', 'test_traj_reverse']
    # case_name = 'test_traj_normal'
    # case_name = 'test_traj_reverse'
    for c in cases:
        gen = DataGen(c)
        gen.generator()