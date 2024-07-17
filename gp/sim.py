import os
import pandas as pd
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gp import GP
import torch
import types
import numpy as np
from scipy.interpolate import splrep, splev


Script_Root = os.path.abspath(os.path.dirname(__file__))


class Simulator:

    def __init__(self, case_name="test_traj"):
        self.data_type = torch.float32
        self.s = 0  # arc length at the start position
        self.x = torch.tensor([[0,0,0,0]]).to(self.data_type)
        self.sim_time = 6  # second
        self.controller_freq = 90  # Hz
        self.data_root = os.path.join(Script_Root, "DATA", case_name)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(1e-3, 0.2)).to(self.data_type)
        self.model = GP(train_x=None, train_y=None, likelihood=self.likelihood).to(self.data_type)

        self.kappa_of_s = self._load_path()
        self.his = []

    def _load_path(self):
        path = os.path.join(self.data_root, "gp_model.pth")
        check_point = torch.load(path)
        self.likelihood.load_state_dict(check_point['likelihood_state_dict'])
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        print(f"loading from check point: {path}")

        df = pd.read_csv(os.path.join(self.data_root, "path.csv"))
        s, curvature = df["s"].values, df["curvature"].values
        tck = splrep(s, curvature)

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
        model = types.SimpleNamespace()
        model.length_front = 0.125  # m
        model.length_rear = 0.125  # m
        model.T = 0.00984

        kappa, n, alpha, v = self.x.numpy()[0]
        v_command, delta =0.6, u.item()

        beta = 0.5*delta
        s_dot = v*np.cos(alpha+beta)/(1-n*kappa)
        n_dot = v*np.sin(alpha+beta)
        alpha_dot = v*np.sin(beta)/model.length_rear - kappa*s_dot
        v_dot = (v_command-v)/model.T
        self.s += s_dot / self.controller_freq
        x_delta =[e/self.controller_freq for e in [n_dot, alpha_dot, v_dot]]
        self.x[0,0] = torch.tensor(self.kappa_of_s(self.s))

        for i in range(0, len(x_delta)):
            self.x[0, i+1] += x_delta[i]


    def start_sim(self):
        iter = self.sim_time * self.controller_freq

        for i in range(iter):
            u, lower, upper = self.predict(self.x)
            self.update(u)
            x = list(self.x.numpy()[0])
            self.his.append([x[0], self.s] + x[1:] + [0.6,u.item()])
            # print(self.his[-1])

        self.save_data()

    def save_data(self):
        heads = ['kappa', 's', 'n', 'alpha', 'v', "v_comm", "delta"]
        data = np.array(self.his)
        df = pd.DataFrame(data=data, columns=heads)
        df.to_csv(os.path.join(self.data_root, 'gp_sim_results.csv'), index=False)

        print(f"saving simulation results at {self.data_root}")

if __name__ == '__main__':
    simulator = Simulator()
    simulator.start_sim()