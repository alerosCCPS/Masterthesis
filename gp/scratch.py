import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

# 示例数据
np.random.seed(42)
data = np.random.rand(100, 3) * 10  # 随机数据
e = np.random.rand(100) * 10  # 示例误差数据

# 创建一个图形窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建一个自定义的颜色映射：从绿色到红色
colors = [(0, 1, 0), (1, 0, 0)]  # 从绿色到红色
n_bins = 100  # 控制颜色渐变的平滑度
cmap_name = 'green_to_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# 归一化 e 数据的绝对值
norm = plt.Normalize(np.min(e), np.max(e))  # 使用 e 的实际范围进行归一化

# 使用 colormap 映射 e 的绝对值到颜色
mapped_colors = cm(norm(e))

# 绘制三维散点图
sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=e, cmap=cm, norm=norm, marker='o')

# 添加颜色条
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=50)
cbar.set_label('Error Magnitude')

# 设置轴标签和标题
ax.set_xlabel('Curvature')
ax.set_ylabel('N')
ax.set_zlabel('Alpha')
ax.set_title("Error on Test Dataset")

# 调整布局以避免颜色条遮挡坐标轴
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

plt.show()
