import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.interpolate import splrep, splev
import pandas as pd
import numpy as np
Script_Root = os.path.abspath(os.path.dirname(__file__))


class SimPlotter:

    def __init__(self,path_name='circle', data_scale=410):
        self.data_scale = data_scale
        self.casename = path_name
        self.data_path = os.path.join(Script_Root, "DATA", 'backUp',path_name)
        self.s, self.n, self.alpha, self.kappa_on_curve = [], [], [], []
        self.load_sim_results()
        self.path_x, self.path_y, self.s_limit, self.init_psi = [], [], [], []
        self.psi_on_curve = []
        self.x_ini, self.y_ini = 0,0
        self.load_path_file()

    def check_data(self, data):
        if len(data) > self.data_scale:
            return data[:self.data_scale]
        return data

    def load_sim_results(self):
        df = pd.read_csv(os.path.join(self.data_path, 'real_results.csv'))
        self.s, self.n, self.alpha,  = self.check_data(df['s'].values), self.check_data(df['n'].values), self.check_data(df['alpha'].values)
        path_df = pd.read_csv(os.path.join(self.data_path, "path.csv"))
        self.kappa_on_curve = splev(self.s, splrep(path_df['s'].values, path_df['curvature'].values))
    def load_path_file(self):
        df = pd.read_csv(os.path.join(self.data_path, 'path.csv'))
        self.path_x = df['x'].values
        self.path_y = df['y'].values
        self.psi_on_curve = splev(self.s, splrep(df['s'].values, df["psi_curve"].values))
        self.x_ini = splev(self.s[0], splrep(df['s'].values, df['x'].values))
        self.y_ini = splev(self.s[0], splrep(df['s'].values, df['y'].values))
        self.s_limit = df['s'].values[-1]
        self.init_psi = self.psi_on_curve[0]


    def cal_XY(self):
        psi = self.init_psi
        x, y = [self.x_ini], [self.y_ini]
        x_c, y_c = self.x_ini, self.y_ini

        for i in range(1, len(self.s)):
            ds = self.s[i] - self.s[i-1]
            ds = ds + self.s_limit if ds < -1 else ds
            psi += ds*self.kappa_on_curve[i]

            x_c += ds * np.cos(psi)
            y_c += ds * np.sin(psi)

            x.append(x_c - self.n[i] * np.sin(psi))
            y.append(y_c + self.n[i] * np.cos(psi))

        return x[1:], y[1:]

    def plot_traj(self):
        px, py = self.cal_XY()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.path_x, self.path_y, linewidth=2, color='black', linestyle="--", alpha=0.3)
        ax.plot(px, py, linewidth=1, color='blue')
        ax.scatter(px[0], py[0], color='green',s=18)
        ax.scatter(px[-1], py[-1], color='black', s=18)
        ax.set_title(self.casename)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        plt.savefig(os.path.join(self.data_path, "sim_traj.png"))
        plt.show()
        print("saving trajectory in: ", self.data_path)

        #  *****************************************************************************
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.path_x, self.path_y, linewidth=2, color='black', linestyle="--", alpha=0.3)
        ax.scatter(px[0], py[0], color='green',s=18)
        ax.scatter(px[-1], py[-1], color='black', s=18)
        twoslope_norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
        cmap = plt.get_cmap("coolwarm")
        for i in range(len(px) - 1):
            ax.plot(px[i:i+2], py[i:i+2], linewidth=1, color=cmap(twoslope_norm(self.n[i])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=twoslope_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='n (m)')
        plt.title("n on test_traj")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig(os.path.join(self.data_path, "n_on_test_traj.png"))
        plt.show()
        print("saving trajectory in: ", self.data_path)
        #  *****************************************************************************

        #  *****************************************************************************
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.path_x, self.path_y, linewidth=2, color='black', linestyle="--", alpha=0.3)
        ax.scatter(px[0], py[0], color='green', s=18)
        ax.scatter(px[-1], py[-1], color='black', s=18)
        twoslope_norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
        cmap = plt.get_cmap("coolwarm")
        for i in range(len(px) - 1):
            ax.plot(px[i:i+2], py[i:i+2], linewidth=1, color=cmap(twoslope_norm(self.alpha[i])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=twoslope_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='alpha (rad)')
        plt.title("alpha on test_traj")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig(os.path.join(self.data_path, "alpha_on_test_traj.png"))
        plt.show()
        print("saving trajectory in: ", self.data_path)
        #  *****************************************************************************


class ResultePlotter:

    def __init__(self, casename='circle', data_scale=410):
        self.data_scale = data_scale
        self.data_root = os.path.join(Script_Root, 'DATA','backUp', casename)
        self.df = pd.read_csv(os.path.join(self.data_root, 'real_results.csv'))

    def check_data(self, data):
        if len(data) > self.data_scale:
            return data[:self.data_scale]
        return data


    def plot(self):
        path_df = pd.read_csv(os.path.join(self.data_root, "path.csv"))
        self.kappa_on_curve = splev(self.df['s'].values, splrep(path_df['s'].values, path_df['curvature'].values))
        s, n, alpha, v = self.check_data(self.kappa_on_curve), \
            self.check_data(self.df['n'].values), self.check_data(self.df['alpha'].values), self.check_data(self.df['v'].values)
        v_command, delta = self.check_data(self.df['v_comm'].values), self.check_data(self.df['delta'].values)
        time = self.check_data(self.df['time'].values)

        x = np.linspace(0, len(s), len(s))
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].plot(x, s, linewidth=2)
        ax[0].set_ylabel("kappa", fontsize=12)
        ax[0].tick_params(axis='both', which='major', labelsize=12)
        ax[0].set_title("kappa")

        ax[1].plot(x, n, linewidth=2)
        ax[1].set_ylabel("n (m)", fontsize=12)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        ax[1].set_title("n")
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax[2].plot(x, alpha, linewidth=2)
        ax[2].set_ylabel("alpha (rad)", fontsize=12)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        ax[2].set_title("alpha")

        ax[3].plot(x, v, linewidth=2)
        ax[3].set_ylabel("v (m/s)", fontsize=12)
        ax[3].tick_params(axis='both', which='major', labelsize=12)
        ax[3].set_title("v")

        plt.subplots_adjust(wspace=0.3)
        fig.text(0.5, 0.02, 'Step', ha='center', fontsize=12)

        plt.suptitle("Trajectory states", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_root, "X_opt.png"), dpi=300)
        plt.show()

        x_u = np.linspace(0, len(v_command), len(v_command))
        fig_u, ax_u = plt.subplots(1, 3, figsize=(18, 4))
        ax_u[0].plot(x_u, v_command, linewidth=2)
        ax_u[0].set_ylabel("v_command (m/s)", fontsize=14)
        ax_u[0].tick_params(axis='both', which='major', labelsize=12)

        ax_u[1].plot(x_u, delta, linewidth=2)
        ax_u[1].set_ylabel("delta (rad)", fontsize=14)
        ax_u[1].tick_params(axis='both', which='major', labelsize=12)

        ax_u[2].plot(x_u, time, linewidth=2)
        ax_u[2].set_ylabel("time (s)", fontsize=14)
        ax_u[2].tick_params(axis='both', which='major', labelsize=12)

        plt.suptitle("Control Law", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_root, "U_opt.png"), dpi=300)
        plt.show()

if __name__ == '__main__':
    # path_name = 'test_traj_mpc'
    path_name = 'test_traj_mpc_adapted'
    # path_name = 'test_traj_mpc_constant'
    # path_name = 'test_traj_mpc_simple'
    # path_name = 'val_traj_mpc_simple'
    # path_name = 'val_traj_mpc_fine'
    data_scale = 800
    plo = SimPlotter(path_name, data_scale)
    plo.plot_traj()
    replot = ResultePlotter(path_name,data_scale)
    replot.plot()