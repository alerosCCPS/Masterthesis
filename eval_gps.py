from gp.path_gp.sim_3D import Simulator as Sim3D
from gp.path_gp.sim_2D import Simulator as Sim2D
from gp.path_gp.sim_3D_LF import Simulator as Sim3DLF
from gp.path_gp.sim_2D_LF import Simulator as Sim2DLF
from utils import ReSampler, Criteria
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

Script_Root = os.path.abspath(os.path.dirname(__file__))

class GPEval:

    def __init__(self):
        self.root_path = Script_Root
        self.datalist = ["train_data_5", "train_data_10","train_data_25","train_data_33","train_data_50", "train_data_66", "train_data_75", "train_data_80", "train_data"]
        self.gp_list = ["2D", "2D_LF", "3D", "3D_LF"]
        self.sim_list = {"2D":Sim2D, "2D_LF":Sim2DLF, "3D":Sim3D, "3D_LF": Sim3DLF}
        self.traj_list = ["test_traj"]
        self.columns = ["GP_type"] + self.datalist
        self.row_title =[f"{traj}_{gp_type}" for traj in self.traj_list for gp_type in self.gp_list]
        self.data = []
        self.results = []

    def sim(self, gp, tra):
        case_name = f"{tra}_{gp}"

        for data_file in self.datalist:
            cost = 0
            for flag in [True, False]:
                simulator = self.sim_list[gp](case_name=case_name, data_file=data_file, reversed=flag)
                simulator.start_sim()
                data_path = os.path.join(self.root_path, "gp", "path_gp","DATA", case_name, "gp_sim_results.csv")
                cost += Criteria(data_path=data_path).cost
            cost = 0.5*cost
            self.data.append(cost)


    def eval(self):
        for tra in self.traj_list:
            for gp in self.gp_list:
                self.sim(gp,tra)
                self.results.append(self.data)
                print(f"tested {tra}_{gp}: ", self.data)
                self.data = []
        self.save_results()

    def save_results(self):
        results = np.vstack(self.results)
        df = pd.DataFrame(results, index=self.row_title, columns=self.columns[1:])
        df.index.name = self.columns[0]
        df.to_csv(os.path.join(self.root_path, "eval_gps.csv"))


def plotter():
    fig, ax = plt.subplots(figsize=(8,6))
    for chunk in pd.read_csv(os.path.join(Script_Root,"eval_gps.csv"), chunksize=1):
        labels = list(chunk.columns)[1:]

        row_data = chunk.iloc[0].values
        case_name, values = row_data[0], row_data[1:]
        x = [i for i in range(len(values))]
        ax.plot(x, values, linewidth=1, label=case_name)
    labels = list(map(lambda s: s.split("_")[-1], labels))[:-1] + ["100"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("GP with different data scale")
    ax.set_xlabel("percentage %")
    ax.set_ylabel("cost")
    # plt.savefig("gp_dataScale_test.png", dpi=300)
    plt.savefig("gp_dataScale_test.pdf", bbox_inches = 'tight', pad_inches = 0.01)
    plt.show()


if __name__ == '__main__':
    # resampler = ReSampler()
    # resampler.process_data()
    gp_evaluator = GPEval()
    gp_evaluator.eval()
    plotter()
