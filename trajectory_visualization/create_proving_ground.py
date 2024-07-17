import os
import matplotlib.pyplot as plt
from utils import check_path
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.interpolate import splrep, splev

from create_trajectory import OCP_Traj
from typing import Optional
import sys

Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(Script_Root, ".."))
R2Deg = lambda x: 180 * x / np.pi


class PathSim:

    def __init__(self, path: SimpleNamespace, name='newGround'):
        self.save_root = os.path.join(Script_Root, 'DATA', 'path', name)
        check_path(self.save_root)
        self.path = path
        self.show_control = True
        self.x = []
        self.y = []
        self.theta = 0

    def load_path(self, path):
        self.path = path

    def init_XY(self):
        if not self.path:
            print("path is empty")
            return
        self.x = np.zeros_like(self.path.arc)
        self.y = np.zeros_like(self.path.arc)
        self.x[0], self.y[0] = self.path.initPos
        self.theta = self.path.initTheta

    def cal_XY(self):
        self.init_XY()
        for index in range(1, self.path.steps):
            ds = self.path.arc[index] - self.path.arc[index - 1]
            self.theta += ds * self.path.curvature[index - 1]
            self.x[index] = self.x[index - 1] + ds * np.cos(self.theta)
            self.y[index] = self.y[index - 1] + ds * np.sin(self.theta)

    def save_path(self):
        data = {"s": self.path.arc,
                "curvature": self.path.curvature,
                # "phi_curve": self.path.phi_curve,
                "initial_pathTheta": self.path.initTheta,
                "terminal_pathTheta": self.path.terminalTheta,
                "x": self.x,
                "y": self.y}
        print(len(data['s']))
        print(len(data['curvature']))
        # print(len(data['phi_curve']))
        print((data['initial_pathTheta']))
        print((data['terminal_pathTheta']))
        print(len(data['x']))
        print(len(data['y']))
        dataFrame = pd.DataFrame(data)
        dataFrame.to_csv(os.path.join(self.save_root, 'path.csv'), index=False)

    def show_path(self):
        self.cal_XY()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.x, self.y, linewidth=2, color='black')
        if len(self.path.bezier_points) and self.show_control:
            ax.scatter(self.path.bezier_points[:, 0],
                       self.path.bezier_points[:, 1],
                       color='green', alpha=0.6)

        ax.scatter(self.x[0], self.y[0],
                   color='blue', alpha=0.6)
        ax.annotate(f'Slope:{round(R2Deg(self.path.initTheta), 1)}',
                    (self.x[0], self.y[0]),
                    textcoords="offset points", xytext=(18, -15),
                    color='blue', ha='center', fontsize=12)

        ax.scatter(self.x[-1], self.y[-1],
                   color='red', alpha=0.6)
        ax.annotate(f'Slope:{round((R2Deg(self.path.terminalTheta)), 1)}',
                    (self.x[-1], self.y[-1]),
                    textcoords="offset points", xytext=(-18, -15),
                    color='red', ha='center', fontsize=12)

        ax.axis('equal')
        ax.set_title("Proving Ground", fontsize=18)
        ax.set_xlabel("X(m)", fontsize=14)
        ax.set_ylabel("Y(m)", fontsize=14)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.savefig(os.path.join(self.save_root, 'path_plot.png'), dpi=300)
        plt.show()
        self.save_path()
        print("save path in folder: ", self.save_root)


class SimplePathPlot:
    def __init__(self, case_name):
        self.data_path = os.path.join(Script_Root, 'DATA', 'path', case_name)
        self.df = pd.read_csv(os.path.join(self.data_path, 'path.csv'))

    def plot(self):
        s = self.df['s']
        curvature = self.df['curvature']
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(s, curvature, linewidth=2)
        ax.set_ylabel("curvature", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.subplots_adjust(wspace=0.3)
        fig.text(0.5, 0.02, 's', ha='center', fontsize=14)

        # plt.suptitle("arc length Path", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_path, "param_path.png"), dpi=300)
        plt.show()


class TrajSim:
    def __init__(self, traj: SimpleNamespace, casename="OCP_Traj"):
        self.traj = traj
        self.casename = casename
        self.path_of_path = self.traj.path_of_path
        path_dataframe = pd.read_csv(self.path_of_path)
        self.path_s = path_dataframe["s"]
        self.path_curvature = path_dataframe["curvature"]
        self.path_phi = path_dataframe["phi_curve"]
        self.path_initTheta = path_dataframe["initial_pathTheta"]
        self.path_termTheta = path_dataframe["terminal_pathTheta"]
        self.path_x = path_dataframe["x"]
        self.path_y = path_dataframe["y"]

        case_name = os.path.basename(os.path.dirname(self.path_of_path))
        self.save_root = os.path.join(Script_Root, 'DATA', 'traj', case_name)
        check_path(self.save_root)
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.theta = 0

    def init_XY(self):
        self.x = np.zeros(self.traj.X_opt.shape[1])
        self.y = np.zeros(self.traj.X_opt.shape[1])
        tck = splrep(self.path_s, self.path_phi, k=3)
        phi_curve = splev(self.traj.X_opt[0, :], tck)
        return phi_curve

    def cal_XY(self):
        phi_curve = self.init_XY()

        x_c, y_c = self.path_x[0], self.path_y[0]
        s = self.traj.X_opt[0]
        n = self.traj.X_opt[1]
        phi_c = phi_curve[0]
        self.x[0] = (x_c - n[0] * np.sin(phi_c))
        self.y[0] = (y_c + n[0] * np.cos(phi_c))

        for i in range(1, self.traj.steps + 1):
            ds = s[i] - s[i-1]
            x_c += ds * np.cos(phi_curve[i])
            y_c += ds * np.sin(phi_curve[i])
            self.x[i] = (x_c - n[i] * np.sin(phi_curve[i]))
            self.y[i] = (y_c + n[i] * np.cos(phi_curve[i]))

    def save_traj(self):
        X_opt = {
            "s": self.traj.X_opt[0],
            "n": self.traj.X_opt[1],
            "alpha": self.traj.X_opt[2],
            "v": self.traj.X_opt[3],
            "traj_x": self.x,
            "traj_y": self.y
        }
        U_opt = {
            "v_command": self.traj.U_opt[0],
            "delta": self.traj.U_opt[1],
        }
        df_state = pd.DataFrame(X_opt)
        df_state.to_csv(os.path.join(self.save_root, 'X_opt.csv'), index=False)
        df_control = pd.DataFrame(U_opt)
        df_control.to_csv(os.path.join(self.save_root, 'U_opt.csv'), index=False)


    def show_traj(self):
        self.cal_XY()
        path_dataframe = pd.read_csv(self.path_of_path)
        path_x, path_y = path_dataframe["x"], path_dataframe["y"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(path_x, path_y, linewidth=2, color='black', linestyle="--", alpha=0.3)
        ax.plot(self.x, self.y, linewidth=2, color='blue')
        ax.set_title(self.casename)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        plt.savefig(os.path.join(self.save_root, "traj_plot.png"))
        plt.show()
        print("saving trajectory in: ", self.save_root)
        self.save_traj()

class SimpleTrajPlot:

    def __init__(self, casename='circle'):
        self.traj_root = os.path.join(Script_Root, 'DATA', 'traj', casename)
        self.df_X = pd.read_csv(os.path.join(self.traj_root, 'X_opt.csv'))
        self.df_U = pd.read_csv(os.path.join(self.traj_root, 'U_opt.csv'))

    def plot(self):
        s, n, alpha, v = self.df_X['s'], self.df_X['n'], self.df_X['alpha'], self.df_X['v']
        v_command, delta = self.df_U['v_command'], self.df_U['delta']

        x = np.linspace(0, len(s), len(s))
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].plot(x, s, linewidth=2)
        ax[0].set_ylabel("s (m)", fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=12)
        # ax[0].set_title("s")

        ax[1].plot(x, n, linewidth=2)
        ax[1].set_ylabel("n (m)", fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        # ax[1].set_title("n")

        ax[2].plot(x, alpha, linewidth=2)
        ax[2].set_ylabel("alpha (rad)", fontsize=14)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        # ax[2].set_title("alpha")

        ax[3].plot(x, v, linewidth=2)
        ax[3].set_ylabel("v (m/s)", fontsize=14)
        ax[3].tick_params(axis='both', which='major', labelsize=12)
        # ax[3].set_title("v")

        plt.subplots_adjust(wspace=0.3)
        fig.text(0.5, 0.02, 'Step', ha='center', fontsize=14)

        plt.suptitle("Trajectory states", fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.traj_root, "X_opt.png"), dpi=300)
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
        plt.savefig(os.path.join(self.traj_root, "U_opt.png"), dpi=300)
        plt.show()


if __name__ == "__main__":
    # plotter = SimplePathPlot('circle')
    # plotter.plot()
    traj_creator = OCP_Traj()
    traj = traj_creator.process_single_path()
    traj_simulator = TrajSim(traj=traj)
    traj_simulator.show_traj()
    traj_plotter = SimpleTrajPlot()
    traj_plotter.plot()

