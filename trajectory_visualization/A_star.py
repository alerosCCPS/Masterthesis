from hamster_dynamic import Hasmster
import numpy as np


def hybrid_a_star(start, goal, alpha_min, alpha_max, v_max, curvature):
    # 定义启发函数
    def heuristic(state, goal):
        # 这里可以使用欧几里得距离作为启发函数
        return np.linalg.norm(np.array(state[:2]) - np.array(goal[:2]))

    # 定义状态转移函数
    def state_transition(state, control, dt):
        # 使用系统动态方程进行状态转移
        s, n, alpha, v = state
        v_command, delta = control
        beta = np.arctan2(Hasmster.length_rear * np.tan(delta), Hasmster.length_front + Hasmster.length_rear)
        s_dot = v * np.cos(alpha + beta) / (1 - n * curvature)
        n_dot = v * np.sin(alpha + beta)
        alpha_dot = v * np.sin(beta) / Hasmster.length_rear - curvature * s_dot
        v_dot = (v_command - v) / Hasmster.T

        s_next = s + s_dot * dt
        n_next = n + n_dot * dt
        alpha_next = alpha + alpha_dot * dt
        v_next = v + v_dot * dt

        return [s_next, n_next, alpha_next, v_next]

    # 定义代价函数
    def cost_function(path):
        # 这里可以根据路径长度或其他因素来定义代价函数
        return len(path)

    # 初始化起点和终点
    open_set = [(start, 0, heuristic(start, goal))]  # (state, cost, heuristic)
    closed_set = []

    # 开始搜索
    while open_set:
        # 从开放列表中选择具有最小总代价的状态
        current_state, current_cost, _ = min(open_set, key=lambda x: x[1] + x[2])
        open_set.remove((current_state, current_cost, heuristic(current_state, goal)))

        # 如果当前状态接近目标状态，则返回路径
        if np.linalg.norm(np.array(current_state[:2]) - np.array(goal[:2])) < 1e-3:
            return current_state

        # 生成当前状态周围的候选状态
        for control in generate_controls(current_state, alpha_min, alpha_max, v_max):
            next_state = state_transition(current_state, control, 0.1)  # 这里的0.1可以根据实际情况调整

            # 检查是否在闭集中或者不符合约束条件
            if next_state in closed_set:
                continue

            # 计算候选状态的总代价
            next_cost = current_cost + 1  # 这里的1可以根据实际情况调整
            heuristic_cost = heuristic(next_state, goal)
            total_cost = next_cost + heuristic_cost

            # 如果候选状态不在开放列表中，则添加到开放列表中
            if next_state not in [item[0] for item in open_set]:
                open_set.append((next_state, next_cost, heuristic_cost))
            # 如果候选状态已经在开放列表中，并且新的代价更低，则更新代价
            else:
                for i, item in enumerate(open_set):
                    if item[0] == next_state and item[1] > next_cost:
                        open_set[i] = (next_state, next_cost, heuristic_cost)

        # 将当前状态添加到闭集中
        closed_set.append(current_state)

    # 如果搜索失败，则返回空路径
    return None


# 生成控制
def generate_controls(state, alpha_min, alpha_max, v_max):
    v_commands = np.linspace(0, v_max, 10)  # 这里的10可以根据实际情况调整
    deltas = np.linspace(alpha_min, alpha_max, 10)  # 这里的10可以根据实际情况调整
    controls = []
    for v_command in v_commands:
        for delta in deltas:
            controls.append([v_command, delta])
    return controls

# 起点和终点
start = [0, 0, 0, 0]
goal = [100, 0, 0, 10]

# 转向角和速度的约束条件
alpha_min = -np.pi / 4
alpha_max = np.pi / 4
v_max = 5

# 路径的弧长和曲率
s = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
curvature = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 执行混合A*算法
path = hybrid_a_star(start, goal, alpha_min, alpha_max, v_max, curvature)

print("最优路径：", path)