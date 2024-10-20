import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

files = {    "splited_velocity_02_06.csv": 0.008977210514987193,
    "splited_velocity_06_02.csv": 0.01045677692256401,
    "splited_velocity_idle_02_06.csv": 0.009902352898816847,
    "splited_velocity_idle_06_02.csv": 0.010023389670210905}
counter = 1
# for f, T in files.items():
#     df = pd.read_csv(f)
#     v = df["velocity"].values
#     time = df["time_steps"].values
#     K = df["K"].values[0]
#     plt.plot(time, v, label="measurement")
#
#     v_init = v[0]
#     v_fitted = v_init+ K*(1-np.exp(-time/T))
#     plt.plot(time, v_fitted, label="fitted")
#
#     plt.xlabel("Time (s)")
#     plt.ylabel("Response")
#     plt.legend()
#     plt.savefig(f"v{counter}.pdf", bbox_inches='tight', pad_inches=0.05)
#     plt.show()
#     counter +=1

T=0.5
K=1
time = np.arange(0,5,0.01)
y = K*(1-np.exp(-time/T))
plt.plot(time, y)
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.title("Step response of first order system")
plt.savefig("step_response.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()


# mean = 0
# variance = 0.1
# noise = np.random.normal(mean, variance, y.shape)
# noised = y+noise
# plt.plot(time, noised, label="noised data")
# b, a = signal.butter(4, 5/(0.5*100), 'low')
# denoise = signal.filtfilt(b, a, noised)
# plt.plot(time, denoise, label="denoise")
# plt.xlabel("Time (s)")
# plt.ylabel("Response")
# plt.title("Step response of first order system")
# plt.legend()
# plt.savefig("sim_nose.pdf", bbox_inches='tight', pad_inches=0.05)
# plt.show()