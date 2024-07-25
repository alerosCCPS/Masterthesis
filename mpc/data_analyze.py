import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
Script_Root = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(Script_Root, "DATA", 'test_traj')


df = pd.read_csv(os.path.join(data_path, 'sim_results.csv'))
curvature = df['curvature'].values
delta = df['delta'].values


diff = delta - curvature

x = [i for i in range(len(curvature))]
plt.plot(x, diff)
plt.title("diff")
plt.savefig(os.path.join(data_path, 'diff.png'))
plt.show()

quotient = delta / curvature
plt.plot(x, quotient)
plt.title("quotient")
plt.savefig(os.path.join(data_path, 'quotient.png'))
plt.show()
mask = abs(quotient)<1
masked = quotient[mask]
print(f"average quotient: {sum(masked)/len(masked)}")

rec_diff = delta - 0.2359*curvature
plt.plot(x, rec_diff)
plt.title("rec_diff")
plt.savefig(os.path.join(data_path, 'rec_diff.png'))
plt.show()