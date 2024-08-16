import numpy as np
import os

import pandas as pd

Script_Root = os.path.abspath(os.path.dirname(__file__))

factor_n = 1e-2
factor_alpha = 0
factor_u = 1e-8
traj_list = ["test_traj"]
controller_list = {"mpc":['mpc'],
                   "gp":['2D', '3D', '2D_LF', '3D_LF']}
columns = ["traj_name"] + controller_list['mpc'] + controller_list['gp']
row_title = traj_list
data = []
def cal_cost(data_path):
    df = pd.read_csv(data_path)
    n = df['n'].to_numpy()
    alpha = df['alpha'].to_numpy()
    delta = df['delta'].to_numpy()
    d_delta = np.gradient(delta)
    return sum(abs(n)*factor_n) + sum(abs(alpha)*factor_alpha) + sum(abs(d_delta)*factor_u)


def eval_tra(con_category, con_typ):
    for tr in traj_list:
        data_path = os.path.join(Script_Root, f"{con_category}", "DATA", f"{tr}_{con_typ}", f'{con_category}_sim_results.csv')
        print(data_path)
        v = cal_cost(data_path)
        print(v)
        data.append(v)
def eval_con():
    for con_category in controller_list.keys():
        for con_type in controller_list[con_category]:
            eval_tra(con_category, con_type)
eval_con()
data = np.array(data).reshape(len(row_title), len(columns)-1)
print(data)
df = pd.DataFrame(data, index=row_title, columns=columns[1:])
df.index.name = columns[0]
df.to_csv("eval.csv")