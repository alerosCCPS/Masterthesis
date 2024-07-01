import casadi as ca
import numpy as np
import pandas as pd
from visualization import SimPlotter, ResultePlotter
from hamster_dynamic import get_hamster_model
from ekf import EKF
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))


class MPC:

    def __init__(self, path_name='circle'):
        self.save_root = os.path.join(Script_Root, "DATA", path_name)
        self.EKF = EKF(path_name)
        self.hamster, self.constrains = get_hamster_model()
        self.sim_time, self.controller_freq = 14, 120  # second, Hz
        self.sample_time = 0.1
        self.N = 20
        self.nu = 2
        self.nx = 4
        self.X = ca.MX.sym("X", self.nx, self.N + 1)
        self.U = ca.MX.sym("U", self.nu, self.N)
        self.P = ca.MX.sym("P", 2 * self.nx)
        self.Q = np.diag([10, 1e-8, 1e-8, 1e-8])
        # self.QN = np.diag([1, 1e-1, 1e-1, 1e-1])
        self.QN = np.diag([10, 1, 1e-8, 1e-8])
        self.R = np.diag([1e-3, 50])
        self.J = 0
        self.g = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.U_opt = []
        self.X_opt = []
        self.setup_nlp()

    def setup_nlp(self):
        # initial constrain
        self.g.append(self.X[:, 0] - self.P[0:self.nx])

        con_ref = np.array([0, 0])
        # system dynamic constrain
        for i in range(self.N):
            state, con = self.X[:, i], self.U[:, i]
            x_error = state - self.P[self.nx:]
            con_diff = con - con_ref
            self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error) + \
                      ca.mtimes(ca.mtimes(con_diff.T, self.R), con_diff)
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
        self.ubx[0:self.nx * (self.N + 1):4] = [self.constrains.s_limit] * len(self.ubx[0:self.nx * (self.N + 1):4])

        # constrain n
        self.lbx[1:self.nx * (self.N + 1):4] = [-self.constrains.n_limit] * len(self.lbx[1:self.nx * (self.N + 1):4])
        self.ubx[1:self.nx * (self.N + 1):4] = [self.constrains.n_limit] * len(self.ubx[1:self.nx * (self.N + 1):4])

        # constrain alpha
        self.lbx[2: self.nx * (self.N + 1):4] = [-self.constrains.alpha_limit] * len(self.lbx[2: self.nx * (self.N + 1):4])
        self.ubx[2: self.nx * (self.N + 1):4] = [self.constrains.alpha_limit] * len(self.ubx[2: self.nx * (self.N + 1):4])

        # constrain v
        self.lbx[3: self.nx * (self.N + 1): 4] = [0] * len(self.lbx[3: self.nx * (self.N + 1): 4])
        self.ubx[3: self.nx * (self.N + 1): 4] = [self.constrains.v_limit] * len(self.ubx[3: self.nx * (self.N + 1): 4])

        # constrain v_comm
        self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
        self.ubx[self.nx * (self.N + 1):: 2] = [self.constrains.v_comm_limit] * len(self.ubx[self.nx * (self.N + 1):: 2])

        # constrain delta
        self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constrains.delta_limit] * len(self.lbx[self.nx * (self.N + 1) + 1:: 2])
        self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constrains.delta_limit] * len(self.ubx[self.nx * (self.N + 1) + 1:: 2])

    def sim(self):
        x_init = [0, 0, 0, 0]
        x_terminal = [self.constrains.s_limit, 0, 0, 0]
        self.X_opt.append(np.array(x_init))
        self.U_opt.append(np.zeros(2))

        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-6}
        nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J,
               'g': self.g, 'p': self.P}
        solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)

        sim_iter = self.sim_time * self.controller_freq
        x0 = ca.repmat(0, self.nx * (self.N + 1) + self.nu * self.N, 1)
        p_values = x_init + x_terminal

        for i in range(sim_iter):
            res = solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p_values)

            x0 = res["x"]
            # states = ca.reshape(x0[0: self.nx * (self.N + 1)], self.nx, self.N + 1).full()
            con = ca.reshape(x0[self.nx * (self.N + 1):], self.nu, self.N).full()

            states_next = p_values[0:self.nx] + self.hamster.dynamic(p_values[0:self.nx], con[:, 0]) / self.controller_freq

            p_values[0:self.nx] = states_next.full()[:, 0]

            x_prior = p_values[0:self.nx]
            u_prior = con[:, 0]

            #  adding noise to simulate measured data
            p_values[0:self.nx] += np.random.normal(0, 0.01, self.nx)

            z_prior = p_values[0:self.nx]

            # Kalman filter
            x_posterior = self.EKF.process(x=x_prior, u=u_prior, z=z_prior)
            p_values[0:self.nx] = x_posterior

            self.X_opt.append(states_next.full()[:, 0])
            self.U_opt.append((con[:, 0]))

        self.save_data()

    def save_data(self):
        data = np.hstack((self.X_opt, self.U_opt))
        headers = ['s', 'n', 'alpha', 'v', 'v_comm', 'delta']
        df = pd.DataFrame(data=data, columns=headers)
        df.to_csv(os.path.join(self.save_root, 'sim_results.csv'), index=False)


if __name__ == "__main__":
    mpc = MPC()
    mpc.sim()
    plo = SimPlotter()
    plo.plot_traj()
    replot = ResultePlotter()
    replot.plot()