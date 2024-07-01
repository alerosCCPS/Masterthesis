import os
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd
import numpy as np
Script_Root = os.path.abspath(os.path.dirname(__file__))


class SimPlotter:

    def __init__(self,path_name='circle'):
        self.casename = path_name
        self.data_path = os.path.join(Script_Root, "DATA", path_name)
        self.s, self.n = self.load_sim_results()
        self.path_x, self.path_y, self.phi_on_curve = self.load_path_file()

    def load_sim_results(self):
        df = pd.read_csv(os.path.join(self.data_path, 'sim_results.csv'))
        return df['s'].values, df['n'].values

    def load_path_file(self):
        df = pd.read_csv(os.path.join(self.data_path, 'path.csv'))
        tck = splrep(df['s'].values, df["phi_curve"].values)
        return df['x'].values, df['y'].values, splev(self.s, tck)

    def cal_XY(self):
        x, y = [self.path_x[0]], [self.path_y[0]]
        x_c, y_c = self.path_x[0], self.path_y[0]

        for i in range(1, len(self.s)):
            ds = self.s[i] - self.s[i-1]
            x_c += ds * np.cos(self.phi_on_curve[i])
            y_c += ds * np.sin(self.phi_on_curve[i])
            x.append(x_c - self.n[i] * np.sin(self.phi_on_curve[i]))
            y.append(y_c + self.n[i] * np.cos(self.phi_on_curve[i]))
        return x, y

    def plot_traj(self):
        px, py = self.cal_XY()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.path_x, self.path_y, linewidth=2, color='black', linestyle="--", alpha=0.3)
        ax.plot(px, py, linewidth=2, color='blue')
        ax.set_title(self.casename)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        plt.savefig(os.path.join(self.data_path, "sim_traj.png"))
        plt.show()
        print("saving trajectory in: ", self.data_path)

class ResultePlotter:

    def __init__(self, casename='circle'):
        self.data_root = os.path.join(Script_Root, 'DATA', casename)
        self.df = pd.read_csv(os.path.join(self.data_root, 'sim_results.csv'))

    def plot(self):
        s, n, alpha, v = self.df['s'], self.df['n'], self.df['alpha'], self.df['v']
        v_command, delta = self.df['v_comm'], self.df['delta']

        x = np.linspace(0, len(s), len(s))
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].plot(x, s, linewidth=2)
        ax[0].set_ylabel("s (m)", fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=12)
        ax[0].set_title("s")

        ax[1].plot(x, n, linewidth=2)
        ax[1].set_ylabel("n (m)", fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        ax[1].set_title("n")

        ax[2].plot(x, alpha, linewidth=2)
        ax[2].set_ylabel("alpha (rad)", fontsize=14)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        ax[2].set_title("alpha")

        ax[3].plot(x, v, linewidth=2)
        ax[3].set_ylabel("v (m/s)", fontsize=14)
        ax[3].tick_params(axis='both', which='major', labelsize=12)
        ax[3].set_title("v")

        plt.subplots_adjust(wspace=0.3)
        fig.text(0.5, 0.02, 'Step', ha='center', fontsize=14)

        plt.suptitle("Trajectory states", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_root, "X_opt.png"), dpi=300)
        plt.show()

        x_u = np.linspace(0, len(v_command), len(v_command))
        fig_u, ax_u = plt.subplots(1, 2, figsize=(10, 4))
        ax_u[0].plot(x_u, v_command, linewidth=2)
        ax_u[0].set_ylabel("v_command (m/s)", fontsize=14)
        ax_u[0].tick_params(axis='both', which='major', labelsize=12)

        ax_u[1].plot(x_u, delta, linewidth=2)
        ax_u[1].set_ylabel("delta (m/s)", fontsize=14)
        ax_u[1].tick_params(axis='both', which='major', labelsize=12)

        plt.suptitle("Trajectory control", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_root, "U_opt.png"), dpi=300)
        plt.show()

if __name__ == '__main__':
    replot = ResultePlotter()
    replot.plot()