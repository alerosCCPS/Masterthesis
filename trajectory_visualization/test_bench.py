from create_path import PathCreator
from create_trajectory import OCP_Traj
from create_proving_ground import PathSim, SimplePathPlot, TrajSim

import numpy as np


def path_creating_circle():

    name = "circle"
    path_creator = PathCreator()
    path = path_creator.create_circle()
    path_simulator = PathSim(path=path,name=name)
    path_simulator.show_path()
    plotter = SimplePathPlot(name)
    plotter.plot()

def path_creating_bezier():
    points = {
        # "path1": np.array([[0,0],[1,0],[1.5,2],[1.5,4]]),  # init_theta=0, terminal_theta=90
        # "path2": np.array([[0, 0], [0, 2], [2, 2], [2, 4]]),  # init_theta=90, terminal_theta=0 , S-form
        # "path3": np.array([[0, 0], [0, 1.5], [1, 4], [2, 4]]),  # init_theta=90, terminal_theta=0
        # "path4": np.array([[0, 0], [1.5, 1.5], [1.5, 2], [0, 3.5]]),  # init_theta=45, terminal_theta=135
        # "path5": np.array([[0, 0], [1, 0], [1, 4], [2, 4]]),  # init_theta=0, terminal_theta=0
        "arc_up" : np.array([[1.25, 0], [1.25, 1], [0.1, 2],[-0.1, -0.3], [-0.75, 0.5], [-1.25,0]]),  # init_theta=90, terminal_theta=90
        "arc_down": np.array([[-1.25, 0], [-2.25, -1], [-0.6, -3.5], [-0.2, 0.4], [1.25, -1.7], [1.25, 0]])  # init_theta=90, terminal_theta=90
    }
    for name, p in points.items():
        path_creator = PathCreator()
        path = path_creator.create_bezier(p)
        path_simulator = PathSim(path=path, name=name)
        path_simulator.show_path()
        plotter = SimplePathPlot(name)
        plotter.plot()

if __name__ =="__main__":
    # path_creating_circle()
    path_creating_bezier()
    # traj_creator = OCP_Traj()
    # traj = traj_creator.process_single_path()
    # traj_simulator = TrajSim(traj=traj)
    # traj_simulator.show_traj()
