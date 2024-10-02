import numpy as np
import os
import pandas as pd
from utils import Criteria
import matplotlib.pyplot as plt
Script_Root = os.path.abspath(os.path.dirname(__file__))


class Eval:

    def __init__(self, sim=True):
        self.sim = sim
        self.root_path = Script_Root
        self.traj_list = [
            "test_traj",
            # 'val_traj'
                          ]
        self.controller_list = {"mpc":['mpc', 'mpc_simple', 'mpc_refine'],
                                "gp":['2D', '3D',
                                      # '2D_LF', '3D_LF'
                                      ]}
        self.columns = ["traj_name"] + self.controller_list['mpc'] + self.controller_list['gp']
        self.row_title = self.traj_list
        self.data = []
        self.results = []

    def eval_controller(self,traj):
        for con_category in self.controller_list.keys():
            for con_type in self.controller_list[con_category]:
                if self.sim:
                    data_path = os.path.join(Script_Root, f"{con_category}", "DATA", f"{traj}_{con_type}",
                                         f'{con_category}_sim_results.csv')
                else:
                    data_path = os.path.join(Script_Root, f"{con_category}", "ros2_ws", "src", f"{con_category}",
                                             f"{con_category}", "DATA","backUp", f"{traj}_{con_type}",
                                             "real_results.csv")
                v = Criteria(data_path).cost
                self.data.append(v)

    def save_results(self):
        results = np.vstack(self.results)
        df = pd.DataFrame(results, index=self.row_title, columns=self.columns[1:])
        df.index.name = self.columns[0]
        filename = "eval_sim.csv" if self.sim else "eval_real.csv"
        df.to_csv(filename)

    def eval(self):
        for traj in self.traj_list:
            self.eval_controller(traj)
            self.results.append(self.data)
            data = [round(i,3) for i in self.data]
            print(f"tested {traj}: ", data)
            self.data = []
        self.save_results()

def plotter(file_name):
    fig, ax = plt.subplots(figsize=(8,6))
    for chunk in pd.read_csv(os.path.join(Script_Root,file_name), chunksize=1):
        labels = list(chunk.columns)[1:]

        row_data = chunk.iloc[0].values
        case_name, values = row_data[0], row_data[1:]
        x = [i for i in range(len(values))]
        ax.plot(x, values, linewidth=1, label=case_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Performance of different controller")
    ax.set_xlabel("controller type")
    ax.set_ylabel("cost")
    plt.savefig(f"{file_name.split('.')[0]}.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    sim = True
    evaluator = Eval(sim=sim)
    evaluator.eval()
    filename = "eval_sim.csv" if sim else "eval_real.csv"
    plotter(filename)

