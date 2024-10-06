import os
import pandas as pd
import matplotlib.pyplot as plt
import math

Script_Root = os.path.abspath(os.path.dirname(__file__))
deg2R = lambda x: math.pi*(x/180)

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
        values_ = []
        for d in self.delta_list:
            # if d<15:
            #     # values_.append(0.15+2.1*math.sin(deg2R(d)*0.5)/0.125)
            #     values_.append(2.1 * math.tan(deg2R(d)) / 0.25)
            # else:
            #     values_.append(2.0)
            values_.append(2.2*math.tanh(deg2R(5*d)))

        plt.plot([i for i in range(len(values_))], values_, label=f'rectified value')
        plt.xlabel("Delta")
        plt.xticks([i for i in range(len(values))], self.delta_list)

        plt.ylabel("Curvature")
        plt.title("Steering Calibration")
        plt.legend()
        plt.savefig(os.path.join(self.data_root, "steering_calib.jpg"))
        plt.show()


if __name__ == "__main__":
    # print(math.sin())
    pro = Process()
    pro.process()
    # r2Deg = lambda x: 180 * x / math.pi
    # k1, k2 = 2, 6
    # L=0.25
    # delta1=25
    # delta2 = r2Deg(k1*L*math.tanh(k2*deg2R(delta1)))
    # print(delta2)