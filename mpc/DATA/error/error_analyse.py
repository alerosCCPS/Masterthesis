import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.interpolate import splrep, splev

df = pd.read_csv("path.csv")
def f(value):
    return splev(value, splrep(df["s"].values, df["curvature"].values, k=3))

df_gt = pd.read_csv("gt.csv")
gt = f(df_gt.iloc[:, 0].values)

df_values = pd.read_csv("sim.csv")
values = f(df_values.iloc[:,0].values)

error = values-gt
X = [i for i in range(1,len(error)+1)]
plt.plot(gt, label="ground truth")
plt.plot(values, label="simulation")

plt.xlabel("prediction horizon")
plt.ylabel("steering")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.savefig("compare.pdf", bbox_inches='tight',pad_inches=0.01)
plt.show()

plt.plot(np.abs(error), label="error")
plt.xlabel("prediction horizon")
plt.ylabel("steering")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.savefig("accumulate.pdf", bbox_inches='tight',pad_inches=0.01)
plt.show()