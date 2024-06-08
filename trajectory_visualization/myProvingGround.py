import os
import matplotlib.pyplot as plt
from utils import check_path
import numpy as np
import pandas as pd
from myPath import MyPath
import sys

Script_Root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(Script_Root, ".."))
R2Deg = lambda x: 180*x/np.pi

class MyPlayGround:

    def __init__(self, path:MyPath, name='newGround'):
        self.save_root = os.path.join(Script_Root, 'DATA','path', name)
        check_path(self.save_root)
        self.path = path
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
            ds = self.path.arc[index] - self.path.arc[index-1]
            self.theta += ds * self.path.curvature[index-1]
            self.x[index] = self.x[index-1] + ds * np.cos(self.theta)
            self.y[index] = self.y[index-1] + ds * np.sin(self.theta)

    def show_path(self):
        self.cal_XY()
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(self.x, self.y, linewidth=2, color='black')
        if len(self.path.bezier_points):
            ax.scatter(self.path.bezier_points[:-1,0],
                       self.path.bezier_points[:-1,1],
                       color='blue', alpha=0.6)
            # mark target point red
            ax.scatter(self.path.bezier_points[-1,0],
                       self.path.bezier_points[-1,1],
                       color='red', alpha=0.6)

            ax.annotate(f'Slope:{round(R2Deg(self.path.initTheta), 1)}',
                        (self.path.bezier_points[0,0], self.path.bezier_points[0,1]),
                        textcoords="offset points", xytext=(18, -15),
                        color='blue', ha='center', fontsize=12)

            ax.annotate(f'Slope:{round((R2Deg(self.path.terminalTheta)),1)}',
                        (self.path.bezier_points[-1,0], self.path.bezier_points[-1,1]),
                        textcoords="offset points", xytext=(-18, -15),
                        color='blue', ha='center', fontsize=12)

        ax.axis('equal')
        ax.set_title("Proving Ground",fontsize=18)
        ax.set_xlabel("X(m)", fontsize=14)
        ax.set_ylabel("Y(m)",fontsize=14)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.savefig(os.path.join(self.save_root, 'path_plot.png'), dpi=300)
        plt.show()
        self.save_path()
        print("save path in folder: ",self.save_root)

    def save_path(self):
        data = {"s": self.path.arc,
                "curvature": self.path.curvature,
                "initial_pathTheta": self.path.initTheta,
                "terminal_pathTheta": self.path.terminalTheta,
                "x": self.x,
                "y": self.y}
        dataFrame = pd.DataFrame(data)
        dataFrame.to_csv(os.path.join(self.save_root, 'path.csv'), index=False)

class SimplePlot:
    def __init__(self, case_name):
        self.data_path = os.path.join(Script_Root, 'DATA','path', case_name)
        self.df = pd.read_csv(os.path.join(self.data_path, 'path.csv'))

    def plot(self):
        s= self.df['s']
        curvature = self.df['curvature']
        x = np.linspace(0,len(s),len(s))
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].plot(x, s, linewidth=2)
        ax[0].set_ylabel("arc length", fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=12)

        ax[1].plot(x, curvature, linewidth=2)
        ax[1].set_ylabel("curvature", fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=12)

        plt.subplots_adjust(wspace=0.3)
        fig.text(0.5, 0.02, 'Step', ha='center', fontsize=14)

        plt.suptitle("Parametrized Path",fontsize=16, y=0.95)
        plt.savefig(os.path.join(self.data_path, "param_path.png"), dpi=300)
        plt.show()

if __name__ == "__main__":
    plotter = SimplePlot('normal')
    plotter.plot()