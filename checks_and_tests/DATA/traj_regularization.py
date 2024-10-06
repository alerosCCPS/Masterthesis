import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))
sample_step = 30

def sample( d):
    return d[10:-10:sample_step]
traj_name = "val_traj"
try:
    # df = pd.read_csv(os.path.join(Script_Root, f"{traj_name}.csv"), sep=";",skiprows=2)
    df = pd.read_csv(os.path.join(Script_Root, f"{traj_name}.csv"))
    # data = df.iloc[0:,0:5]
    data = df

    x_c, y_c = data['x_m'].values, data['y_m'].values
    original_kappa = data["kappa"].values

    # x_c, y_c = np.flip(x_c), np.flip(y_c)
    x_c, y_c = y_c, -x_c

    s_c = data['s_m'].values
    start_index = 0
    x_c = np.hstack([x_c[start_index:], x_c[1:start_index+1]])
    y_c = np.hstack([y_c[start_index:], y_c[1:start_index+1]])
    s_c = np.hstack([s_c[start_index:]-s_c[start_index], s_c[1:start_index+1]+s_c[-1]-s_c[start_index]])

    x_padding = np.hstack([x_c[-3:-1], x_c, x_c[1:3]])
    y_padding = np.hstack([y_c[-3:-1], y_c, y_c[1:3]])
    dx = np.gradient(x_padding)
    dy = np.gradient(y_padding)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    psi = np.arctan2(dy, dx)
    curvature = -(ddx * dy - ddy * dx) / (dx ** 2 + dy ** 2) ** 1.5

    psi = psi[2:-2]
    curvature = curvature[2:-2]
    # curvature = original_kappa
    for i in range(len(curvature)):
        if curvature[i]>2.5:
            curvature[i] = curvature[i-1] if i-1>=0 else 0
        elif curvature[i]<-2.5:
            curvature[i] = curvature[i-1] if i-1>=0 else 0
    # curvature[-1] = curvature[-2] + (curvature[-2]-curvature[-3])
    # steps = 14
    # delta = (0.7766061406646112 - curvature[-1])/steps
    # for i in range(14):
    #     curvature[i] = curvature[-1] + i*delta


    # regu_data = []
    # for idx in ['# s_m', ' x_m', ' y_m', ' kappa_radpm']:
    #     regu_data.append(data[idx].values)
    # for i in [' psi_rad']:
    #     trans = 0.5 * np.pi + data[i].values
        # temp = trans[-1]
        # trans[1:] = trans[0:-1]
        # trans[0] = temp
        # trans[trans > 2*np.pi] %= 2*np.pi
        # trans[trans < -2 * np.pi] %= -2 * np.pi
        # trans[trans > np.pi] -= 2*np.pi
        # trans[trans < -np.pi] += 2*np.pi
        # regu_data.append(trans)
    s_c = sample(s_c)
    x_c = sample(x_c)
    y_c = sample(y_c)
    psi = sample(psi)
    curvature = sample(curvature)

    regu_data = [s_c, x_c, y_c,psi, curvature, [0]*len(s_c), [0]*len(s_c)]
    for i in range(1, len(s_c)):
        if s_c[i]<=s_c[i-1]:
            print(f"s not increasing at({i},{i-1})" )
    regu_data = np.column_stack(regu_data)
    columns = ["s", 'x', 'y', 'psi_curve', 'curvature', "v", "a"]
    data_save = pd.DataFrame(regu_data, columns=columns)
    if not os.path.exists(os.path.join(Script_Root, f'{traj_name}_mpc')):
        os.makedirs(os.path.join(Script_Root, f'{traj_name}_mpc'))
    data_save.to_csv(os.path.join(Script_Root, f'{traj_name}_mpc', 'path.csv'), index=False)
    data_save.to_csv(os.path.join(Script_Root, 'test_traj.csv'),sep=';', index=False)


    # x,y = data_save['x'].values, data_save['y'].values
    # s, kappa = data_save['s'].values, data_save['curvature'].values
    x,y,s,kappa = x_c, y_c, s_c, curvature
    norm = plt.Normalize(kappa.min(), kappa.max())
    cmap = plt.get_cmap('coolwarm')
    fig, ax = plt.subplots()
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(kappa[i])))

    # plt.plot(x, y)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Curvature (kappa)')
    plt.title("val_traj")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig(os.path.join(Script_Root, f"{traj_name}_mpc", "path_plot.png"))
    plt.show()

    max_curvature = [2]*len(s)
    min_curvature = [-2] * len(s)
    plt.plot(s, kappa)
    # plt.plot(s, max_curvature, color='r')
    # plt.plot(s, min_curvature, color='r')
    plt.xlabel('s (m)')
    plt.ylabel("curvature")
    plt.savefig(os.path.join(Script_Root, f"{traj_name}_mpc", "param_path.png"))
    plt.show()

except FileNotFoundError:
    print(f"file {traj_name} not found ")
