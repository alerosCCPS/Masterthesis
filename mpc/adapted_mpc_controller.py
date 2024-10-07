import casadi as ca
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
import time
from visualization import SimPlotter, ResultePlotter
from hamster_dynamic import get_hamster_model, load_path
from ekf import EKF
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))
deg2R = lambda x: ca.pi*(x/180)
r2Deg = lambda x: 180*x/np.pi

class MPC:

    def __init__(self, path_name='test_traj'):
        self.threshold_delta = 25
        self.No_track = False
        self.kappa_ref = 0
        self.x_init = [0, 0.0, 0, 0]  #s, n, alpha, v
        # self.x_init = [0, 0, -0.07496703, 0.6]  #s, n, alpha_ref[0], v_ref

        self.path_name = path_name
        self.interpolator, _, self.s_max = load_path(self.path_name)
        self.save_root = os.path.join(Script_Root, "DATA", path_name)
        # self.EKF = EKF(path_name)
        self.hamster, self.constraints = get_hamster_model(path_name)

        self.sim_time, self.controller_freq = 18, 60  # second, Hz

        self.sample_time = 0.1
        self.N = 20
        self.nu = 2
        self.nx = 4
        self.simple_mode = True
        self.X = ca.MX.sym("X", self.nx, self.N + 1)
        self.U = ca.MX.sym("U", self.nu, self.N)
        self.P = ca.MX.sym("P", 2 * self.nx)

        # self.Q = np.diag([0., 5e2, 1e-1, 1e1])
        # self.QN = np.diag([0., 1e3, 1e-1, 1e1])
        # self.R = np.diag([1e-3, 1e-3, 1e1, 1e1])  # v_error, delta_error, v_diff, delta_diff

        self.Q = np.diag([0, 1, 1, 1e-1])
        self.QN = np.diag([0, 1, 1, 1e-1])
        self.R = np.diag([1, 1, 1e-1, 1e-1])

        self.J = 0
        self.g = []  # currently forced to zero later (system dynamic constraints; multiple shooting)
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.U_opt = []
        self.X_opt = []
        # to store more of what is happening; this might seem unnecessary, but for me it was almost always useful to undertsnad what is happening and to make sure my code/ocp makes sense
        self.OCP_results_X = []  #
        self.OCP_results_U = []
        self.Solverstats = []  # store status of solver for each OCP
        self.feasible_iterations = []  # to check if the ocp was always feasible in closed loop

        self.time_list = []
        self.solver = None
        self.setup_nlp()

    def setup_nlp(self):
        # initial constrain
        self.g.append(self.X[:, 0] - self.P[0:self.nx])

        if self.simple_mode:
            s = ca.if_else(self.X[:, 0][0] > self.s_max, self.X[:, 0][0] - self.s_max, self.X[:, 0][0])
            kappa = self.interpolator(s)

        con_pre = np.array([0, 0])
        # system dynamic constrain
        for i in range(self.N):
            state, con = self.X[:, i], self.U[:, i]

            if not self.simple_mode:
                s = ca.if_else(self.X[:, i][0] > self.s_max, self.X[:, i][0] - self.s_max, self.X[:, i][0])
                kappa = self.interpolator(s)
            L =self.hamster.length_front + self.hamster.length_rear
            l_r = self.hamster.length_rear
            alpha_ref = -ca.asin(kappa * l_r)

            # delta_ref = 2*ca.arcsin(kappa*self.hamster.length_rear)
            # delta_ref = ca.arctanh(ca.tan(delta_ref)/(self.hamster.steer_k1*L))/self.hamster.steer_k2
            delta_ref = ca.arctanh(kappa/self.hamster.steer_k1)/self.hamster.steer_k2
            delta_ref = ca.if_else(delta_ref > deg2R(self.threshold_delta), deg2R(self.threshold_delta), delta_ref )
            delta_ref = ca.if_else(delta_ref < -deg2R(self.threshold_delta), -deg2R(self.threshold_delta), delta_ref)
            con_error = con - ca.vertcat(self.P[-1], delta_ref)

            con_diff = con - con_pre
            con_pre = con
            con_join = ca.vertcat(con_error, con_diff)
            x_error = state - ca.vertcat(0,0,alpha_ref,self.P[-1])
            self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error) \
                      + ca.mtimes(ca.mtimes(con_join.T, self.R), con_join)
            state_dot = self.hamster.dynamic(state, con, kappa)
            state_next = state + self.sample_time * state_dot
            self.g.append(state_next - self.X[:, i + 1])  # multiple shooting constraint
        if not self.simple_mode:
            s = ca.if_else(self.X[:, -1][0] > self.s_max, self.X[:, -1][0] - self.s_max, self.X[:, -1][0])
            kappa = self.interpolator(s)
        alpha_ref = -ca.asin(kappa * self.hamster.length_rear)

        x_error = self.X[:, -1] - ca.vertcat(0,0,alpha_ref,self.P[-1])
        self.J += ca.mtimes(ca.mtimes(x_error.T, self.QN), x_error)

        self.g = ca.vertcat(*self.g)
        self.lbg = [0] * self.g.size()[0]
        self.ubg = [0] * self.g.size()[0]
        self.lbx = [-ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)
        self.ubx = [ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)

        # constrain s
        self.lbx[0:self.nx * (self.N + 1):4] = [0] * len(self.lbx[0:self.nx * (self.N + 1):4])
        # self.ubx[0:self.nx * (self.N + 1):4] = [self.constraints.s_limit] * len(self.ubx[0:self.nx * (self.N + 1):4])

        # constrain n
        self.lbx[1:self.nx * (self.N + 1):4] = [-self.constraints.n_limit] * len(self.lbx[1:self.nx * (self.N + 1):4])
        self.ubx[1:self.nx * (self.N + 1):4] = [self.constraints.n_limit] * len(self.ubx[1:self.nx * (self.N + 1):4])

        # constrain alpha
        self.lbx[2: self.nx * (self.N + 1):4] = [-self.constraints.alpha_limit] * len(
            self.lbx[2: self.nx * (self.N + 1):4])
        self.ubx[2: self.nx * (self.N + 1):4] = [self.constraints.alpha_limit] * len(
            self.ubx[2: self.nx * (self.N + 1):4])

        # constrain v
        self.lbx[3: self.nx * (self.N + 1): 4] = [0] * len(self.lbx[3: self.nx * (self.N + 1): 4])
        self.ubx[3: self.nx * (self.N + 1): 4] = [self.constraints.v_limit] * len(
            self.ubx[3: self.nx * (self.N + 1): 4])

        # constrain v_comm
        self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
        self.ubx[self.nx * (self.N + 1):: 2] = [self.constraints.v_comm_limit] * len(
            self.ubx[self.nx * (self.N + 1):: 2])

        # constrain delta
        self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constraints.delta_limit] * len(
            self.lbx[self.nx * (self.N + 1) + 1:: 2])
        self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constraints.delta_limit] * len(
            self.ubx[self.nx * (self.N + 1) + 1:: 2])

        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-6}
        nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J,
               'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)

        print("set up nlp solver")

    def predict(self, x0=ca.repmat(0, 4 * (20 + 1) + 2 * 20, 1), x=[0, 0, 0, 0]):
        p = x + [0, 0, 0, self.constraints.targ_vel]
        start = time.time()
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        cal_time = time.time() - start
        x0_res = res["x"]
        con = ca.reshape(x0_res[self.nx * (self.N + 1):], self.nu, self.N).full()  # planned control input (due to nlp formulation they are stored at the end of the solution vector)

        # store all OCP results
        status = self.solver.stats()  # solver status
        status_flag = True
        if status['return_status'] == 'Infeasible_Problem_Detected':
            print("OCP seems to be infeasible, please check carefully")
            status_flag = False

        U_OCP = con  # planned input sequence
        X_OCP = ca.reshape(x0_res[0:self.nx * (self.N + 1)], self.nx, self.N + 1).full()  # predicted state sequence

        u_star = con[:,0]
        u_star[0] = 0.2 if u_star[0]<0.2 else u_star[0]
        return x0_res, u_star, cal_time, status, U_OCP, X_OCP, status_flag

    def sim(self):

        #  x = [s, n, alpha, v]
        x_init = self.x_init

        sim_iter = int(self.sim_time * self.controller_freq)

        kappa = self.interpolator(x_init[0]).full()
        L =self.hamster.length_front + self.hamster.length_rear
        l_r = self.hamster.length_rear

        x0x = ca.repmat(x_init, (self.N + 1))
        x0u = ca.repmat([self.constraints.v_limit, l_r * kappa], self.N)
        x0 = ca.vertcat(x0x, x0u)

        for i in range(sim_iter):
            self.X_opt.append(x_init)
            x0, con, cal_time, status, U_OCP, X_OCP,_ = self.predict(x0, list(x_init))

            self.time_list.append(cal_time)
            self.OCP_results_X.append(X_OCP)
            self.OCP_results_U.append(U_OCP)
            self.Solverstats.append(status)

            states_next = x_init + self.hamster.dynamic(x_init, con, self.interpolator(x0[0])) / self.controller_freq

            x_init = states_next.full()[:, 0]

            # x_prior = x_init
            # u_prior = con

            #  adding noise to simulate measured data

            # z_prior = x_prior + np.random.normal(0, 0.01, self.nx)
            # noise = np.random.normal(0, 0.01)
            # z_prior = x_prior + np.array([0, noise, 0, 0])
            # # Kalman filter
            # # x_posterior = self.EKF.process(x=x_prior, u=u_prior, z=z_prior)
            # # x_init = x_posterior
            # x_init = z_prior

            if x_init[0] >= self.constraints.s_limit:
                print("start next round ", x_init[0])
                x_init[0] -= self.constraints.s_limit

            self.U_opt.append(con)
        print("finished simulation")
        print(f"average cal time: {sum(self.time_list) / len(self.time_list)}")
        print(f"the worst case: {max(self.time_list)}")

        # was everything feasible?
        self.feasible_iterations = [
            False if self.Solverstats[n]['return_status'] == 'Infeasible_Problem_Detected' else True for n in range(
                len(self.Solverstats))]  # maybe there are some cases, where a different solver stat also means infeasible; so take the "true" statement here with care
        if not all(self.feasible_iterations):
            print("there was an infeasible problem. Please check carefully.")
        else:
            print('everything seemed to be feasible')
        self.save_data()

    def save_data(self):
        path_df = pd.read_csv(os.path.join(self.save_root, "path.csv"))
        data = np.hstack((self.X_opt, self.U_opt, np.array(self.time_list).reshape(-1, 1)))
        kappa = splev(data[:, 0], splrep(path_df["s"].values, path_df["curvature"]))
        headers = ["curvature", 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'time_list']
        data = np.hstack((np.array(kappa).reshape(-1, 1), data))
        df = pd.DataFrame(data=data, columns=headers)
        df.to_csv(os.path.join(self.save_root, 'mpc_sim_results.csv'), index=False)


if __name__ == "__main__":

    # path_name = 'val_traj_mpc_adapted'
    # path_name = 'val_traj_mpc_adapted_simple'
    # path_name = 'test_traj_mpc_adapted_simple'
    # path_name = 'test_traj_mpc_adapted'
    for p in [
        # 'val_traj_mpc_adapted',
        # 'val_traj_mpc_adapted_simple',
        'test_traj_mpc_adapted_simple',
        # 'test_traj_mpc_adapted'
    ]:
        mpc = MPC(p)
        mpc.sim()
        plo = SimPlotter(p)
        plo.plot_traj()
        replot = ResultePlotter(p)
        replot.plot()