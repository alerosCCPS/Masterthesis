import casadi as ca

# 定义系统动态方程
def dynamics(x, u):
    A = ca.MX.zeros(2, 2)
    A[0, 1] = 1
    B = ca.MX.zeros(2, 1)
    B[1, 0] = 1
    return ca.mtimes(A, x) + ca.mtimes(B, u)

# 定义目标函数和约束
def objective_constraint(T, X, U, N, x0, xf, u_min, u_max):
    J = 0
    g = []
    dt = T / N
    for k in range(N-1):
        x_current, u_current = X[k*2:(k+1)*2], U[k]
        k1 = dynamics(x_current, u_current)
        k2 = dynamics(x_current + dt/2 * k1, u_current)
        k3 = dynamics(x_current + dt/2 * k2, u_current)
        k4 = dynamics(x_current + dt * k3, u_current)
        x_next = x_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        J += dt
        g.append(X[2*(k+1):2*(k+2)] - x_next)
    g.append(X[2*N:] - xf)
    g.append(U - ca.repmat(ca.vertcat(u_min), N, 1))
    g.append(ca.repmat(ca.vertcat(u_max), N, 1) - U)
    return J, g

# 参数
x0 = [0, 0]  # 初始状态
xf = [1, 0]  # 目标状态
u_min = -1
u_max = 1
N = 20  # 时间步数

# 创建符号变量
X = ca.MX.sym('X', 2*(N+1), 1)
print("shape of X", X.shape)
U = ca.MX.sym('U', N, 1)
T = ca.MX.sym('T')

# 构建目标函数和约束
J, g = objective_constraint(T, X, U, N, x0, xf, u_min, u_max)

# 构建优化问题
nlp = {'x': ca.vertcat(ca.reshape(X, 2*(N+1), 1), ca.reshape(U, N, 1), T),
       'f': J,
       'g': ca.vertcat(*g)}
opts = {'ipopt.print_level': 0,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-6,
        'ipopt.acceptable_obj_change_tol': 1e-6,
        'ipopt.max_iter': 1000}

# 创建求解器
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# 初始猜测
x_guess = ca.vertcat(ca.reshape(ca.DM.zeros(2, N+1), 2*(N+1), 1),
                      ca.DM.zeros(N, 1),
                      10.0)

# 求解问题
sol = solver(x0=x_guess)

# 提取结果
x_opt = sol['x'][:2*(N+1)].full().reshape(2, N+1)
u_opt = sol['x'][2*(N+1):2*(N+1)+N].full().reshape(1, N)
T_opt = sol['x'][-1]

print(f"Optimal time: {T_opt}")
print(f"Optimal state trajectory:\n{x_opt}")
print(f"Optimal control trajectory:\n{u_opt}")
