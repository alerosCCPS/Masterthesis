# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:27:13 2024

test_new_ocp
"""

# import rclpy
# from rclpy.node import Node
# from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
# from hamster_interfaces.msg import TrackingState
# from std_msgs.msg import Header, Bool, Float32
from scipy.interpolate import splev, splrep
from visualization2 import SimPlotter, ResultePlotter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from hamster_dynamic_nobeta import get_hamster_model, load_path

import casadi as ca
import numpy as np
import pandas as pd
# from scipy.interpolate import splev, splrep
import time
# from visualization import SimPlotter, ResultePlotter
# from ekf import EKF
import types
import json
import os


Script_Root = os.path.abspath(os.path.dirname(__file__))

class MPC_Fine:

    def __init__(self, path_name='test_traj'):
        self.No_track = False
        self.kappa_ref = 1
        self.x_init = [0, 0, 0, 0]
        self.path_name = path_name
        self.interpolator, _, self.s_max = load_path(self.path_name)
        self.save_root = os.path.join(Script_Root, "DATA", path_name)
        # self.EKF = EKF(path_name)
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
        self.Q = np.diag([0.,1, 1, 1])
        self.QN = np.diag([0., 1, 1, 1])

        self.R = np.diag([1, 1])
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
        self.OCP_results_kappa =[]
        self.Solverstats = []  # store status of solver for each OCP
        self.feasible_iterations = []  # to check if the ocp was always feasible in closed loop

        self.time_list = []
        self.x0_ = None
        self.solver = None
        self.setup_nlp()

    def setup_nlp(self):
        # initial constrain
        self.g.append(self.X[:, 0] - self.P[0:self.nx])

        if self.simple_mode and not self.No_track:
            s = ca.if_else(self.X[:, 0][0] > self.s_max, self.X[:, 0][0] - self.s_max, self.X[:, 0][0])
            kappa = self.interpolator(s)
        elif self.simple_mode and self.No_track:
            kappa = self.kappa_ref
            #s = ca.if_else(self.X[:, 0][0] > self.s_max, self.X[:, 0][0] - self.s_max, self.X[:, 0][0])

        # system dynamic constrain
        for i in range(self.N):
            state, con = self.X[:, i], self.U[:, i]

            if not self.simple_mode:
                s = ca.if_else(self.X[:, i][0] > self.s_max, self.X[:, i][0] - self.s_max, self.X[:, i][0])
                kappa = self.interpolator(s)
            L =self.hamster.length_front + self.hamster.length_rear
            l_r = self.hamster.length_rear
            alpha_ref = 0#-ca.asin(kappa * l_r)
            delta_ref = ca.atan(L*kappa)#L/l_r * ca.asin(kappa*l_r)
            con_error = con - ca.vertcat(delta_ref, self.P[-1])
            x_error = state - ca.vertcat(0,0,alpha_ref,self.P[-1])
            self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error) \
                      + ca.mtimes(ca.mtimes(con_error.T, self.R), con_error)
            state_dot = self.hamster.dynamic(state, con, kappa)
            state_next = state + self.sample_time * state_dot
            self.g.append(state_next - self.X[:, i + 1])  # multiple shooting constraint
        if not self.simple_mode:
            s = ca.if_else(self.X[:, -1][0] > self.s_max, self.X[:, -1][0] - self.s_max, self.X[:, 0-1][0])
            kappa = self.interpolator(s)
        alpha_ref = 0#-ca.asin(kappa * l_r)
        x_error = self.X[:, -1] - ca.vertcat(0,0,alpha_ref,self.P[-1])
        self.J += ca.mtimes(ca.mtimes(x_error.T, self.QN), x_error) # terminal cost

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
        
        
        # constraint delta 
        self.lbx[self.nx * (self.N + 1):: 2] = [-self.constraints.delta_limit] * len(self.lbx[self.nx * (self.N + 1):: 2])
        self.ubx[self.nx * (self.N + 1):: 2] = [self.constraints.delta_limit] * len(
            self.ubx[self.nx * (self.N + 1):: 2])
        
        #  constrain v_comm
        self.lbx[self.nx * (self.N + 1) + 1:: 2] = [0] * len(
            self.lbx[self.nx * (self.N + 1) + 1:: 2])
        self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constraints.v_comm_limit]* len(
            self.ubx[self.nx * (self.N + 1) + 1:: 2])
        
        # # constrain v_comm
        # self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
        # self.ubx[self.nx * (self.N + 1):: 2] = [self.constraints.v_comm_limit] * len(
        #     self.ubx[self.nx * (self.N + 1):: 2])

        # # constrain delta
        # self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constraints.delta_limit] * len(
        #     self.lbx[self.nx * (self.N + 1) + 1:: 2])
        # self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constraints.delta_limit] * len(
        #     self.ubx[self.nx * (self.N + 1) + 1:: 2])

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
        x0u = ca.repmat([ca.atan(self.hamster.L*self.kappa_ref), 0.6],self.N)
        x0 = ca.vertcat(x0x,x0u)
        for i in range(sim_iter):
            self.X_opt.append(x_init)
            x0, con, cal_time, status, U_OCP, X_OCP = self.predict(x0, list(x_init))
            self.time_list.append(cal_time)
            self.OCP_results_X.append(X_OCP)
            self.OCP_results_U.append(U_OCP)
            if self.simple_mode and self.No_track:
                kappa_OCP = np.repeat(self.kappa_ref,self.N+1).reshape((self.N+1,1))
            else:
                kappa_OCP = self.interpolator(X_OCP[0,:]).full()
                
            self.OCP_results_kappa.append(kappa_OCP)
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
        self.save_data()

    def save_data(self):
        path_df = pd.read_csv(os.path.join(self.save_root, "path.csv"))
        data = np.hstack((self.X_opt, self.U_opt, np.array(self.time_list).reshape(-1,1)))
        kappa = splev(data[:,0], splrep(path_df["s"].values, path_df["curvature"]))
        headers = ["curvature", 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'time_list']
        data = np.hstack((np.array(kappa).reshape(-1,1), data))
        df = pd.DataFrame(data=data, columns=headers)
        df.to_csv(os.path.join(self.save_root, 'mpc_sim_results.csv'), index=False)
    
    def plot_closed_loop(self):
        #kappa =  np.asarray(self.OCP_results_kappa)[0,:]
        kappa_inits = np.concatenate(self.OCP_results_kappa,1)[0,:]
        s, n, alpha, v = np.asarray(self.X_opt)[:,0], np.asarray(self.X_opt)[:,1], np.asarray(self.X_opt)[:,2], np.asarray(self.X_opt)[:,3]
        delta,v_command, time_list = np.asarray(self.U_opt)[:,0], np.asarray(self.U_opt)[:,1], self.time_list #self.df['v_comm'], self.df['delta'],self.df['time_list']

        x = np.linspace(0, len(s), len(s))
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].plot(x, s, linewidth=2)
        ax[0].set_ylabel("path progress", fontsize=12)
        ax[0].tick_params(axis='both', which='major', labelsize=12)
        ax[0].set_title("path progress")

        ax[1].plot(x, n, linewidth=2)
        ax[1].set_ylabel("n (m)", fontsize=10)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        ax[1].set_title("n")
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

        ax[2].plot(x, alpha, linewidth=2)
        ax[2].set_ylabel("alpha (rad)", fontsize=12)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        ax[2].set_title("alpha")

        ax[3].plot(x, v, linewidth=2)
        ax[3].set_ylabel("v (m/s)", fontsize=12)
        ax[3].tick_params(axis='both', which='major', labelsize=12)
        ax[3].set_title("v")

        plt.subplots_adjust(wspace=0.3)
        fig.text(0.5, 0.02, 'Step', ha='center', fontsize=14)

        plt.suptitle("Trajectory states", fontsize=16, y=0.95)
        #plt.savefig(os.path.join(self.data_root, "X_opt.png"), dpi=300)
        plt.show()

        x_u = np.linspace(0, len(v_command), len(v_command))
        fig_u, ax_u = plt.subplots(1, 3, figsize=(18, 4))
        ax_u[0].plot(x_u, v_command, linewidth=2)
        ax_u[0].set_ylabel("v_command (m/s)", fontsize=14)
        ax_u[0].tick_params(axis='both', which='major', labelsize=12)

        ax_u[1].plot(x_u, delta, linewidth=2)
        ax_u[1].set_ylabel("delta (rad)", fontsize=14)
        ax_u[1].tick_params(axis='both', which='major', labelsize=12)

        ax_u[2].plot(x_u, time_list, linewidth=2)
        ax_u[2].set_ylabel("cal time (s)", fontsize=14)
        ax_u[2].tick_params(axis='both', which='major', labelsize=12)

        plt.suptitle("Control Law", fontsize=16, y=0.95)
        #plt.savefig(os.path.join(self.data_root, "U_opt.png"), dpi=300)
        plt.show()
        
        plt.figure()
        plt.plot(kappa_inits)
        #plt.xlabel('pathprogress')
        plt.xlabel('index')
        plt.ylabel('ref curvature')
        plt.show()
        
    # def plot_track_XY(self):
    #     return 1



if __name__ == "__main__":
    # path_name = 'test_traj_mpc_simple'
    # path_name = 'test_traj_reverse'
    path_name = 'test_traj_mpc'
    # path_name = 'val_traj_mpc'
    # path_name = 'val_traj_mpc_simple'
    mpc = MPC_Fine(path_name)
    mpc.sim()
    mpc.plot_closed_loop()
    
    plo = SimPlotter(path_name)
    plo.plot_traj()
    #replot = ResultePlotter(path_name)
    #replot.plot()