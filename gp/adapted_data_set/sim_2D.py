import sys
import os
Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(Script_Root)

import pandas as pd
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gp_2D import GP
import torch
import types
import numpy as np
from scipy.interpolate import splrep, splev
import math


class Simulator:

    def __init__(self, case_name="test_traj",data_file="train_data", reversed=False):
        self.reversed = reversed
        self.k = 0.25
        self.steer_k1 = 2.25
        self.steer_k2 = 4
        self.length_rear = 0.125
        self.T = 0.00984
        self.L = 0.25
        self.targ_v = 0.4
        self.data_type = torch.float32
        self.s = 0  # arc length at the start position
        self.x = torch.tensor([[0,0,0,0]]).to(self.data_type)
        self.sim_time = 22  # second
        self.max_s = 0
        self.controller_freq = 60  # Hz
        self.data_root = os.path.join(Script_Root, "DATA", case_name)

        self.data_x, self.data_y = None, None
        self._load_data(data_file)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(train_x=self.data_x, train_y=self.data_y, likelihood=self.likelihood).to(self.data_type)

        self.kappa_of_s = self._load_path()
        self.his = []

    def _load_data(self,data_file):
        path = os.path.join(self.data_root, "..", f"{data_file}.csv")
        df = pd.read_csv(path)
        n, alpha, delta, rec_diff = df['n'].values.reshape((-1, 1)), \
            df['alpha'].values.reshape((-1, 1)), \
            df['delta'].values.reshape((-1, 1)), \
            df['rec_diff'].values.reshape((-1, 1))
        self.data_x = torch.tensor(np.hstack((n, alpha))).to(self.data_type)
        self.data_y = torch.tensor(rec_diff).to(self.data_type).squeeze_()

    def _load_path(self):
        path = os.path.join(self.data_root, "gp_2D.pth")
        check_point = torch.load(path)
        self.likelihood.load_state_dict(check_point['likelihood_state_dict'])
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        print(f"loading from check point: {path}")
        if not self.reversed:
            df = pd.read_csv(os.path.join(self.data_root, "path_normal.csv"))
        else:
            df = pd.read_csv(os.path.join(self.data_root, "path_reversed.csv"))

        s, curvature = df["s"].values, df["curvature"].values
        tck = splrep(s, curvature)

        self.max_s = s[-1]
        print(f"max s: {self.max_s}")

        self.x[0,0] = torch.tensor(splev(self.s, tck=tck))
        print(f"initialization state: {self.x}")
        def f(value):
            return splev(value, tck)
        return f

    def predict(self,x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            u = observed_pred.mean.numpy()
            lower, upper = observed_pred.confidence_region()
        return u, lower, upper

    def update(self, u:torch.tensor):

        kappa, n, alpha, v = self.x.numpy()[0]
        v_command, delta =self.targ_v, u.item()
        # delta = self.steer_k * delta + self.L * self.steer_c
        delta = math.atan(self.steer_k1*self.L * math.tanh(self.steer_k2*delta))

        beta = 0.5*delta
        s_dot = v*np.cos(alpha+beta)/(1-n*kappa)
        n_dot = v*np.sin(alpha+beta)
        alpha_dot = v*np.sin(beta)/self.length_rear - kappa*s_dot
        v_dot = (v_command-v)/self.T
        self.s += s_dot / self.controller_freq
        if self.s>=self.max_s:
            self.s -= self.max_s
        x_delta =[e/self.controller_freq for e in [n_dot, alpha_dot, v_dot]]
        self.x[0,0] = torch.tensor(self.kappa_of_s(self.s))

        for i in range(0, len(x_delta)):
            self.x[0, i+1] += x_delta[i]


    def start_sim(self):
        iter = int(self.sim_time * self.controller_freq)

        for i in range(iter):
            x_value = self.x[0,1:3].unsqueeze(0)
            u, lower, upper = self.predict(x_value)
            u +=math.atanh(self.x[0,0].item()/ self.steer_k1)/self.steer_k2
            self.update(u)

            x = list(self.x.numpy()[0])
            self.his.append([x[0], self.s] + x[1:] + [self.targ_v,u.item()])
            # print(self.his[-1])

        self.save_data()

    def save_data(self):
        heads = ['kappa', 's', 'n', 'alpha', 'v', "v_comm", "delta"]
        data = np.array(self.his)
        df = pd.DataFrame(data=data, columns=heads)
        df.to_csv(os.path.join(self.data_root, 'gp_sim_results.csv'), index=False)
        print(f"saving simulation results at {self.data_root}")

if __name__ == '__main__':
    case_name = 'test_traj_2D'#synthetic'
    simulator = Simulator(case_name)
    simulator.start_sim()