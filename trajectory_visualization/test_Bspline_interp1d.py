import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, interp1d

# 生成已知数据点
x = np.linspace(0, 10, 10)
y = np.sin(x) + np.random.normal(0, 0.1, x.size)  # 添加一些噪声

# 三次 B-spline 插值
tck = splrep(x, y, k=3)
x_b_spline = np.linspace(0, 10, 100)
y_b_spline = splev(x_b_spline, tck)

# 三次多项式插值
f_cubic = interp1d(x, y, kind='cubic')
x_cubic = np.linspace(0, 10, 100)
y_cubic = f_cubic(x_cubic)

# 绘制结果
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_b_spline, y_b_spline, label='Cubic B-spline interpolation')
plt.plot(x_cubic, y_cubic, label='Cubic polynomial interpolation')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Cubic B-spline and Cubic Polynomial Interpolation')
plt.show()
