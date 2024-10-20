import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv("path.csv", delimiter=";")
df = pd.read_csv("path.csv")
x, y, s, kappa = df["x"],df["y"],df["s"], df["curvature"]
norm = plt.Normalize(kappa.min(), kappa.max())
cmap = plt.get_cmap('coolwarm')
fig, ax = plt.subplots()
for i in range(len(x) - 1):
    ax.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(kappa[i])))

# plt.plot(x, y)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Curvature')
plt.title("test trajectory")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.savefig("test_path_plot.pdf",bbox_inches='tight',pad_inches=0.01)
plt.show()
#
# max_curvature = [2] * len(s)
# min_curvature = [-2] * len(s)
plt.plot(s, kappa)
# plt.plot(s, max_curvature, color='r')
# plt.plot(s, min_curvature, color='r')
plt.xlabel('s (m)')
plt.ylabel("curvature")
plt.savefig("test_param_path.pdf", bbox_inches='tight',pad_inches=0.01)
plt.show()