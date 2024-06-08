from myPath import MyPath
from myProvingGround import MyPlayGround, SimplePlot
import numpy as np

if __name__ =="__main__":

    path = MyPath()
    path.create_circle()
    # path.create_bezier(np.array([[0,0],[1,0],[1.5,2],[1.5,4]]))
    name = "circle"
    ground = MyPlayGround(path=path,name=name)
    ground.show_path()
    plotter = SimplePlot(name)
    plotter.plot()

    # points = {
    #     "path1": np.array([[0,0],[1,0],[1.5,2],[1.5,4]]),  # init_theta=0, terminal_theta=90
    #     "path2": np.array([[0, 0], [0, 2], [2, 2], [2, 4]]),  # init_theta=90, terminal_theta=0 , S-form
    #     "path3": np.array([[0, 0], [0, 1.5], [1, 4], [2, 4]]),  # init_theta=90, terminal_theta=0
    #     "path4": np.array([[0, 0], [1.5, 1.5], [1.5, 2], [0, 3.5]]),  # init_theta=45, terminal_theta=135
    #     "path5": np.array([[0, 0], [1, 0], [1, 4], [2, 4]]),  # init_theta=0, terminal_theta=0
    #     "test" : np.array([[0, 0], [0, 1.5], [1.5, 0], [2, 2], [2,4]]) # init_theta=90, terminal_theta=90
    # }
    # for name, p in points.items():
    #     path.create_bezier(p)
    #     ground = MyPlayGround(path=path, name=name)
    #     ground.show_path()
    #     plotter = SimplePlot(name)
    #     plotter.plot()
