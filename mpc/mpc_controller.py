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


class MPC:

    def __init__(self, path_name='test_traj'):
        self.No_track = False
        self.kappa_ref = 0
        self.x_init = [0, 0, 0, 0]
        self.path_name = path_name
        self.interpolator, _, self.s_max = load_path(self.path_name)
        self.save_root = os.path.join(Script_Root, "DATA", path_name)
        self.EKF = EKF(path_name)
        self.hamster, self.constraints = get_hamster_model(path_name)
        self.sim_time, self.controller_freq = 16, 60  # second, Hz
        self.sample_time = 0.1
        self.N = 10
        self.nu = 2
        self.nx = 4
        self.simple_mode = False
        self.X = ca.MX.sym("X", self.nx, self.N + 1)
        self.U = ca.MX.sym("U", self.nu, self.N)
        self.P = ca.MX.sym("P", 2 * self.nx) 
        self.Q = np.diag([0., 1e-2, 0, 1e-1])
        self.QN = np.diag([0., 1e-2, 0, 1e-1])
        # self.Q = np.diag([0, 0, 0, 1e-1])
        # self.QN = np.diag([0, 0, 0, 1e-1])
        # self.Q = np.diag([0, 1, 1, 1e-1])
        # self.QN = np.diag([0, 1, 1, 1e-1])
        # self.R = np.diag([1e-1, 1e-1])
        self.R = np.diag([0, 0])
        self.J = 0
        self.g = [] # currently forced to zero later (system dynamic constraints; multiple shooting)
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.U_opt = []
        self.X_opt = []
        # to store more of what is happening; this might seem unnecessary, but for me it was almost always useful to undertsnad what is happening and to make sure my code/ocp makes sense
        self.OCP_results_X = [] #
        self.OCP_results_U = []
        self.Solverstats = [] # store status of solver for each OCP
        self.feasible_iterations = [] # to check if the ocp was always feasible in closed loop
        
        self.time_list = []
        self.solver = None
        self.setup_nlp()

    def setup_nlp(self):
        # initial constrain
        self.g.append(self.X[:, 0] - self.P[0:self.nx])

        con_ref = np.array([0, 0])

        if self.simple_mode and not self.No_track:
            s = ca.if_else(self.X[:, 0][0]>self.s_max, self.X[:, 0][0]-self.s_max, self.X[:, 0][0])
            kappa = self.interpolator(s)
        elif self.simple_mode and self.No_track:
            kappa = self.kappa_ref
            s = ca.if_else(self.X[:, 0][0]>self.s_max, self.X[:, 0][0]-self.s_max, self.X[:, 0][0])

        # system dynamic constrain
        for i in range(self.N):
            state, con = self.X[:, i], self.U[:, i]
            x_error = state - self.P[self.nx:]
            if not self.simple_mode:
                s = ca.if_else(self.X[:, i][0] > self.s_max, self.X[:, i][0] - self.s_max, self.X[:, i][0])
                kappa = self.interpolator(s)

            con_diff = con - con_ref
            con_ref = con
            self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error)\
                      +ca.mtimes(ca.mtimes(con_diff.T, self.R), con_diff)
            state_dot = self.hamster.dynamic(state, con, kappa)
            state_next = state + self.sample_time * state_dot
            self.g.append(state_next - self.X[:, i + 1]) # multiple shooting constraint
        x_error = self.X[:,-1] - self.P[self.nx:]
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
        self.lbx[2: self.nx * (self.N + 1):4] = [-self.constraints.alpha_limit] * len(self.lbx[2: self.nx * (self.N + 1):4])
        self.ubx[2: self.nx * (self.N + 1):4] = [self.constraints.alpha_limit] * len(self.ubx[2: self.nx * (self.N + 1):4])

        # constrain v
        self.lbx[3: self.nx * (self.N + 1): 4] = [0] * len(self.lbx[3: self.nx * (self.N + 1): 4])
        self.ubx[3: self.nx * (self.N + 1): 4] = [self.constraints.v_limit] * len(self.ubx[3: self.nx * (self.N + 1): 4])

        # constrain v_comm
        self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
        self.ubx[self.nx * (self.N + 1):: 2] = [self.constraints.v_comm_limit] * len(self.ubx[self.nx * (self.N + 1):: 2])

        # constrain delta
        self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constraints.delta_limit] * len(self.lbx[self.nx * (self.N + 1) + 1:: 2])
        self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constraints.delta_limit] * len(self.ubx[self.nx * (self.N + 1) + 1:: 2])

        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-6}
        nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J,
               'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)

        print("set up nlp solver")

    def predict(self,x0=ca.repmat(0, 4 * (20 + 1) + 2 * 20, 1), x=[0, 0, 0, 0]): # x0=ca.repmat(0, 4 * (20 + 1) + 2 * 20, 1) is probably no longer correct as you changed the prediction horizon, maybe remove it in the method definition?# also you maybe do not want to always use a zero inital guess
        p = x + [0, 0, 0, 0.6] 
        start = time.time()
        res = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        cal_time = time.time() - start
        x0_res = res["x"]
        con = ca.reshape(x0_res[self.nx * (self.N + 1):], self.nu, self.N).full() # planned control input (due to nlp formulation they are stored at the end of the solution vector)
        
        # store all OCP results
        status = self.solver.stats() # solver status
        if status['return_status'] == 'Infeasible_Problem_Detected':
            print("OCP seems to be infeasible, please check carefully")
            
        U_OCP = con # planned input sequence
        X_OCP = ca.reshape(x0_res[0:self.nx * (self.N + 1)], self.nx, self.N+1).full() #predicted state sequence

        return x0_res, con[:, 0], cal_time, status, U_OCP, X_OCP

    def sim(self):

        #  x = [s, n, alpha, v]
        # x_init = [0, 0.04, 0.05*np.pi, 0]
        x_init = self.x_init#[0, 0, 0, 0] #[0, 0, 0, 0.2]

        sim_iter = int(self.sim_time * self.controller_freq)
        #x0 = ca.repmat(0, self.nx * (self.N + 1) + self.nu * self.N, 1)
        x0x = ca.repmat(x_init,(self.N + 1))
        x0u = ca.repmat([0.6 ,self.hamster.length_front*self.kappa_ref],self.N)
        x0 = ca.vertcat(x0x,x0u)
        for i in range(sim_iter):
            x0, con, cal_time, status, U_OCP, X_OCP = self.predict(x0, list(x_init))
            self.time_list.append(cal_time)
            self.OCP_results_X.append(X_OCP)
            self.OCP_results_U.append(U_OCP)
            self.Solverstats.append(status)
            if self.No_track and self.simple_mode:
                states_next = x_init + self.hamster.dynamic(x_init, con, self.kappa_ref) / self.controller_freq
            else:
                states_next = x_init + self.hamster.dynamic(x_init, con, self.interpolator(x0[0])) / self.controller_freq
            
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

            if x_init[0] >= self.constraints.s_limit:
                print("start next round ",x_init[0])
                x_init[0] -= self.constraints.s_limit

            self.X_opt.append(x_init)
            self.U_opt.append(con)
        print("finished simulation")
        print(f"average cal time: {sum(self.time_list)/len(self.time_list)}")
        print(f"the worst case: {max(self.time_list)}")
        
        # was everything feasible?
        self.feasible_iterations = [False if self.Solverstats[n]['return_status'] == 'Infeasible_Problem_Detected' else True  for n in range(len(self.Solverstats))] # maybe there are some cases, where a different solver stat also means infeasible; so take the "true" statement here with care 
        if not all(self.feasible_iterations):
            print("there was an infeasible problem. Please check carefully.")
        else:
            print('everything seemed to be feasible')    
        #self.save_data()

    def save_data(self):
        path_df = pd.read_csv(os.path.join(self.save_root, "path.csv"))
        data = np.hstack((self.X_opt, self.U_opt, np.array(self.time_list).reshape(-1,1)))
        kappa = splev(data[:,0], splrep(path_df["s"].values, path_df["curvature"]))
        headers = ["curvature", 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'time_list']
        data = np.hstack((np.array(kappa).reshape(-1,1), data))
        df = pd.DataFrame(data=data, columns=headers)
        df.to_csv(os.path.join(self.save_root, 'mpc_sim_results.csv'), index=False)


if __name__ == "__main__":
    # path_name = 'test_traj_mpc_simple'
    # path_name = 'test_traj_reverse'
    # path_name = 'test_traj_mpc'
    path_name = 'val_traj_mpc'
    path_name = 'val_traj_mpc_simple'
    mpc = MPC(path_name)
    mpc.sim()
    plo = SimPlotter(path_name)
    plo.plot_traj()
    replot = ResultePlotter(path_name)
    replot.plot()