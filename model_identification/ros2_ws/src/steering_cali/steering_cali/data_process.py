import os
import pandas as pd
import matplotlib.pyplot as plt

Script_Root = os.path.abspath(os.path.dirname(__file__))

class Process:

    def __init__(self):
        self.data_root = os.path.join(Script_Root, "DATA")
        self.v_list = [i *0.2 for i in range(1,4)]
        self.delta_list = [i*5 for i in range(1,6)]

    def process(self):
        for v in self.v_list:
            values = []
            for delta in self.delta_list:
                data_path = os.path.join(self.data_root, f"velocity0_{str(int(10*v))}_delta_f{str(delta)}.csv")
                df = pd.read_csv(data_path)
                values.append(1/df.values[-1])
            plt.plot([i for i in range(len(values))], values,label=f'Velocity {str(round(v,2))}')
        plt.xlabel("Delta")
        plt.xticks([i for i in range(len(values))], self.delta_list)
        plt.ylabel("Curvature")
        plt.title("Steering Calibration")
        plt.legend()
        plt.savefig(os.path.join(self.data_root, "steering_calib.jpg"))
        plt.show()


if __name__ == "__main__":
    pro = Process()
    pro.process()