import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.interpolate import splrep, splev
import pandas as pd
import numpy as np
Script_Root = os.path.abspath(os.path.dirname(__file__))


class SimPlotter:

    def __init__(self,path_name='test_traj', reversed=False):
        self.reversed = reversed
        self.casename = path_name
        self.data_path = os.path.join(Script_Root, "DATA", path_name)
        self.s, self.n, self.alpha = [], [], []
        self.load_sim_results()
        self.path_x, self.path_y, self.s_limit, self.init_psi = [], [], [], []
        self.psi_on_curve, self.kappa_on_curve = [], []
        self.x_ini, self.y_ini = 0,0
        self.load_path_file()

    def load_sim_results(self):
        df = pd.read_csv(os.path.join(self.data_path, 'gp_sim_results.csv'))
        self.s, self.n, self.alpha = df['s'].values, df['n'].values, df['alpha'].values

    def load_path_file(self):
        if not self.reversed:
            df = pd.read_csv(os.path.join(self.data_path, 'path_normal.csv'))
        else:
            df = pd.read_csv(os.path.join(self.data_path, 'path_reversed.csv'))
        self.path_x = df['x'].values
        self.path_y = df['y'].values
        self.psi_on_curve = splev(self.s, splrep(df['s'].values, df["psi_curve"].values))
        self.kappa_on_curve = splev(self.s, splrep(df['s'].values, df['curvature'].values))
        self.X = splev(self.s, splrep(df['s'].values, df['x'].values))
        self.Y = splev(self.s, splrep(df['s'].values, df['y'].values))
        self.s_limit = df['s'].values[-1]
        # self.init_psi = df['psi_curve'].values[0]
        self.init_psi = self.psi_on_curve[0]


    def cal_XY(self):
        x, y = [self.X[0]], [self.Y[0]]
        psi = self.init_psi
        for i in range(1, len(self.s)):
            ds = self.s[i]-self.s[i-1]
            if ds<-2:
                psi = self.init_psi
            else:
                psi += ds * self.kappa_on_curve[i]

            x.append(self.X[i] - self.n[i] * np.sin(psi))
            y.append(self.Y[i] + self.n[i] * np.cos(psi))

        return x[1:], y[1:]

    def plot_traj(self):
        px, py = self.cal_XY()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.path_x, self.path_y, linewidth=2, color='black', linestyle="--", alpha=0.3)
        ax.plot(px, py, linewidth=1, color='blue')
        ax.scatter(px[0], py[0], color='green',s=18)
        ax.scatter(px[-1], py[-1], color='black', s=18)
        # ax.set_title(self.casename)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        plt.savefig(os.path.join(self.data_path, "gp_sim.png"))
        # plt.show()
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
        plt.savefig(os.path.join(self.data_path, "gp_n_on_traj.png"))
        # plt.show()
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
        plt.savefig(os.path.join(self.data_path, "gp_alpha_on_traj.png"))
        plt.show()
        print("saving trajectory in: ", self.data_path)
        #  *****************************************************************************


class ResultePlotter:
    def __init__(self, casename='test_traj'):
        self.data_root = os.path.join(Script_Root, 'DATA', casename)
        self.df = pd.read_csv(os.path.join(self.data_root, 'gp_sim_results.csv'))

    def plot(self):
        s, n, alpha, v = self.df['kappa'], self.df['n'], self.df['alpha'], self.df['v']
        v_command, delta = self.df['v_comm'], self.df['delta']

        x = np.linspace(0, len(s), len(s))
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].plot(x, s, linewidth=2)
        ax[0].set_ylabel("kappa", fontsize=12)
        ax[0].tick_params(axis='both', which='major', labelsize=12)
        ax[0].set_title("kappa")

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

        # plt.suptitle("Trajectory states", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_root, "gp_X_opt.png"), dpi=300)
        plt.show()

        x_u = np.linspace(0, len(v_command), len(v_command))
        fig_u, ax_u = plt.subplots(1, 2, figsize=(12, 4))
        ax_u[0].plot(x_u, v_command, linewidth=2)
        ax_u[0].set_ylabel("v_command (m/s)", fontsize=14)
        ax_u[0].tick_params(axis='both', which='major', labelsize=12)

        ax_u[1].plot(x_u, delta, linewidth=2)
        ax_u[1].set_ylabel("delta (m/s)", fontsize=14)
        ax_u[1].tick_params(axis='both', which='major', labelsize=12)

        # plt.suptitle("Control Law", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_root, "gp_U_opt.png"), dpi=300)
        plt.show()

if __name__ == '__main__':

    for p in [
        'test_traj_3D',
        'test_traj_2D',
        'test_traj_3D_LF',
        'test_traj_2D_LF',
        'val_traj_3D',
        'val_traj_2D',
        'val_traj_3D_LF',
        'val_traj_2D_LF',
    ]:
        plo = SimPlotter(p)
        plo.plot_traj()
        replot = ResultePlotter(p)
        replot.plot()