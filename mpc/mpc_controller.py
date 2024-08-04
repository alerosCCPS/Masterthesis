import casadi as ca
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
import time
from visualization import SimPlotter, ResultePlotter
from hamster_dynamic import get_hamster_model
from ekf import EKF
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))


class MPC:

    def __init__(self, path_name='test_traj'):
        self.save_root = os.path.join(Script_Root, "DATA", path_name)
        self.EKF = EKF(path_name)
        self.hamster, self.constrains = get_hamster_model(path_name)
        self.sim_time, self.controller_freq = 13.5, 60  # second, Hz
        self.sample_time = 0.1
        self.N = 20
        self.nu = 2
        self.nx = 4
        self.X = ca.MX.sym("X", self.nx, self.N + 1)
        self.U = ca.MX.sym("U", self.nu, self.N)
        self.P = ca.MX.sym("P", 2 * self.nx)
        # self.Q = np.diag([0., 1e-8, 1e-8, 1e-1])
        # self.QN = np.diag([0., 1e-8, 1e-8, 1e-1])
        self.Q = np.diag([0, 0, 0, 1e-1])
        self.QN = np.diag([0, 0, 0, 1e-1])
        # self.R = np.diag([1e-8, 1e-8])
        # self.R = np.diag([0, 0])
        self.J = 0
        self.g = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.U_opt = []
        self.X_opt = []
        self.time_list = []
        self.solver = None
        self.setup_nlp()

    def setup_nlp(self):
        # initial constrain
        self.g.append(self.X[:, 0] - self.P[0:self.nx])

        con_ref = np.array([0, 0])
        # system dynamic constrain
        for i in range(self.N):
            state, con = self.X[:, i], self.U[:, i]
            x_error = state - self.P[self.nx:]

            # con_diff = con - con_ref
            self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error)
                      # ca.mtimes(ca.mtimes(con_diff.T, self.R), con_diff)
            state_dot = self.hamster.dynamic(state, con)
            state_next = state + self.sample_time * state_dot
            self.g.append(state_next - self.X[:, i + 1])
        x_error = self.X[:,-1] - self.P[self.nx:]
        self.J += ca.mtimes(ca.mtimes(x_error.T, self.QN), x_error)

        self.g = ca.vertcat(*self.g)
        self.lbg = [0] * self.g.size()[0]
        self.ubg = [0] * self.g.size()[0]
        self.lbx = [-ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)
        self.ubx = [ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)

        # constrain s
        self.lbx[0:self.nx * (self.N + 1):4] = [0] * len(self.lbx[0:self.nx * (self.N + 1):4])
        # self.ubx[0:self.nx * (self.N + 1):4] = [self.constrains.s_limit] * len(self.ubx[0:self.nx * (self.N + 1):4])

        # constrain n
        self.lbx[1:self.nx * (self.N + 1):4] = [-self.constrains.n_limit] * len(self.lbx[1:self.nx * (self.N + 1):4])
        self.ubx[1:self.nx * (self.N + 1):4] = [self.constrains.n_limit] * len(self.ubx[1:self.nx * (self.N + 1):4])

        # constrain alpha
        self.lbx[2: self.nx * (self.N + 1):4] = [-self.constrains.alpha_limit] * len(self.lbx[2: self.nx * (self.N + 1):4])
        self.ubx[2: self.nx * (self.N + 1):4] = [self.constrains.alpha_limit] * len(self.ubx[2: self.nx * (self.N + 1):4])

        # constrain v
        self.lbx[3: self.nx * (self.N + 1): 4] = [0.2] * len(self.lbx[3: self.nx * (self.N + 1): 4])
        self.ubx[3: self.nx * (self.N + 1): 4] = [self.constrains.v_limit] * len(self.ubx[3: self.nx * (self.N + 1): 4])

        # constrain v_comm
        self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
        self.ubx[self.nx * (self.N + 1):: 2] = [self.constrains.v_comm_limit] * len(self.ubx[self.nx * (self.N + 1):: 2])

        # constrain delta
        self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constrains.delta_limit] * len(self.lbx[self.nx * (self.N + 1) + 1:: 2])
        self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constrains.delta_limit] * len(self.ubx[self.nx * (self.N + 1) + 1:: 2])

        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-6}
        nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J,
               'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)

        print("set up nlp solver")

    def predict(self,x0=ca.repmat(0, 4 * (20 + 1) + 2 * 20, 1), x=[0, 0, 0, 0]):
        p = x + [0, 0, 0, 0.6]
        start = time.time()
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        cal_time = time.time() - start
        x0_res = res["x"]
        con = ca.reshape(x0_res[self.nx * (self.N + 1):], self.nu, self.N).full()

        return x0_res, con[:, 0], cal_time

    def sim(self):

        #  x = [s, n, alpha, v]
        # x_init = [0, 0.04, 0.05*np.pi, 0]
        x_init = [0, 0, 0, 0]

        sim_iter = int(self.sim_time * self.controller_freq)
        x0 = ca.repmat(0, self.nx * (self.N + 1) + self.nu * self.N, 1)

        for i in range(sim_iter):
            x0, con, cal_time = self.predict(x0, list(x_init))
            self.time_list.append(cal_time)

            states_next = x_init + self.hamster.dynamic(x_init, con) / self.controller_freq
            x_init = states_next.full()[:,0]

            # x_prior = x_init
            # u_prior = con
            #
            # #  adding noise to simulate measured data
            #
            # z_prior = x_prior + np.random.normal(0, 0.01, self.nx)
            #
            # # Kalman filter
            # x_posterior = self.EKF.process(x=x_prior, u=u_prior, z=z_prior)
            # x_init = x_posterior

            if x_init[0] >= self.constrains.s_limit:
                print("start next round ",x_init[0])
                x_init[0] -= self.constrains.s_limit

            self.X_opt.append(x_init)
            self.U_opt.append(con)
        print("finished simulation")
        print(f"average cal time: {sum(self.time_list)/len(self.time_list)}")
        print(f"the worst case: {max(self.time_list)}")

        self.save_data()

    def save_data(self):
        path_df = pd.read_csv(os.path.join(self.save_root, "path.csv"))
        data = np.hstack((self.X_opt, self.U_opt, np.array(self.time_list).reshape(-1,1)))
        kappa = splev(data[:,0], splrep(path_df["s"].values, path_df["curvature"]))
        headers = ["curvature", 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'time_list']
        data = np.hstack((np.array(kappa).reshape(-1,1), data))
        df = pd.DataFrame(data=data, columns=headers)
        df.to_csv(os.path.join(self.save_root, 'sim_results.csv'), index=False)


if __name__ == "__main__":
    path_name = 'test_traj_normal'
    # path_name = 'test'
    mpc = MPC(path_name)
    mpc.sim()
    plo = SimPlotter(path_name)
    plo.plot_traj()
    replot = ResultePlotter(path_name)
    replot.plot()