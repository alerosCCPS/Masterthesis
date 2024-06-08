import casadi as ca

# 定义系统动态方程
def dynamics(x, u):
    A = ca.MX.zeros(2, 2)
    A[0, 1] = 1
    B = ca.MX.zeros(2, 1)
    B[1, 0] = 1
    return ca.mtimes(A, x) + ca.mtimes(B, u)

# 定义变量
x = ca.MX.sym('x', 2)  # 状态变量
u = ca.MX.sym('u')     # 控制变量

# 参数
x0 = [0, 0]  # 初始状态
xf = [1, 0]  # 目标状态
u_min = -1
u_max = 1
N = 20  # 时间步数

# 创建优化问题
opti = ca.Opti()

# 定义变量
X = opti.variable(2, N+1)  # 状态变量矩阵
U = opti.variable(1, N)    # 控制变量矩阵
T = opti.variable()        # 优化时间区间

# 初始状态约束
opti.subject_to(X[:, 0] == x0)

# 动态方程约束
dt = T / N
for k in range(N):
    k1 = dynamics(X[:, k], U[:, k])
    k2 = dynamics(X[:, k] + dt/2 * k1, U[:, k])
    k3 = dynamics(X[:, k] + dt/2 * k2, U[:, k])
    k4 = dynamics(X[:, k] + dt * k3, U[:, k])
    x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X[:, k+1] == x_next)

# 终端状态约束
opti.subject_to(X[:, N] == xf)

# 控制信号约束
opti.subject_to(opti.bounded(u_min, U, u_max))

# 定义目标函数：最小化时间
opti.minimize(T)

# 设置初始猜测值
opti.set_initial(X, ca.DM.zeros(2, N+1))
opti.set_initial(U, ca.DM.zeros(1, N))
opti.set_initial(T, 10.0)  # 初步选择的时间区间

# 调整求解器选项
opts = {
    'ipopt.print_level': 0,
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6,
    'ipopt.acceptable_tol': 1e-6,
    'ipopt.acceptable_obj_change_tol': 1e-6
}
opti.solver('ipopt', opts)

# 求解问题
sol = opti.solve()

# 提取结果
x_opt = sol.value(X)
u_opt = sol.value(U)
T_opt = sol.value(T)

print(f"Optimal time: {T_opt}")
print(f"Optimal state trajectory:\n{x_opt}")
print(f"Optimal control trajectory:\n{u_opt}")
