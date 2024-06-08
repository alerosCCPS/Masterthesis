from hamster_dynamic import Hasmster, get_casadi_model
from scipy.interpolate import interp1d
import casadi as ca
import pandas as pd
import numpy as np
from typing import Optional
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))


class ASTAR_Traj:

    def __init__(self):
        self.sys = Hasmster()
        self.path_root = os.path.join(Script_Root, 'DATA')
        self.path_set = os.listdir(self.path_root)
        self.the_path = {
            "s": [],
            "curvature": []
        }
        self.init_state = [0, 0, 0, 0]
        self.terminal_state = [0, 0, 0, 0]  # s,n,alpha,v
        self.s_to_curvature = None  # interpolation function
        self.trajectory = []

    def process_single_case(self, case_name, init_n: float, init_alpha: float):
        df = pd.read_csv(os.path.join(self.path_root, case_name, "path.csv"))
        self.the_path["s"] = df["s"]
        self.the_path["curvature"] = df["curvature"]
        self.s_to_curvature = interp1d(self.the_path["s"], self.the_path["curvature"])
        self.init_state = [0, init_n, init_alpha, 0]
        self.terminal_state = [self.the_path["s"][-1], 0, 0, 0]

    def heuristic(self, state):
        return np.linalg.norm(state[:3] - self.terminal_state[:3]) + np.abs(state[-1] - self.terminal_state[-1])

    def state_transition(self, state, control, dt):
        dstate = self.sys.update(state=state, u=control, curvature=self.s_to_curvature(state[0]))
        return state + dstate * dt

    def a_star(self):
        pass

    def process_batch_mode(self, init_n: float, init_alpha: float):
        for case in self.path_set:
            self.process_single_case(case_name=case, init_n=init_n, init_alpha=init_alpha)


class TIME_OPT_Traj:

    def __init__(self):
        self.model = get_casadi_model()
        self.dt = 0.02  # Time step (second)
        self.N: Optional[int] = 200  # Prediction horizon
        self.nx = 4
        self.nu = 2
        self.path_root = os.path.join(Script_Root, 'DATA', 'path')
        self.path_list = os.listdir(self.path_root)

    def process_single_path(self, case_name='circle'):
        df = pd.read_csv(os.path.join(self.path_root, case_name, 'path.csv'))
        s, kappa = list(df["s"].values), list(df["curvature"].values)

        s_2_kappa = ca.interpolant("s_2_kappa","bspline",[s], kappa)
        T_var = ca.MX.sym('T_var')

        X = ca.MX.sym('X', self.nx * (self.N + 1), 1)  # State sequence
        U = ca.MX.sym('U', self.nu * self.N, 1)  # Input sequence
        xf = [s[-1], 0, 0, 0]

        g = []
        J = 0

        self.dt = T_var/self.N
        # *************************** equal constrain ************************************
        for i in range(self.N):
            x_current, u_current = X[i*self.nx: (i+1)*self.nx], U[i*self.nu: (i+1)*self.nu]
            # print(s_2_kappa(x_current[0]))
            k1 = self.model.dynamic(x=x_current, u=u_current, curvature=1)
            # x_k1 = x_current + 0.5 * self.dt * k1
            # print(s_2_kappa(x_k1[0]))
            # k2 = self.model.dynamic(x=x_k1, u=u_current, curvature=s_2_kappa(x_k1[0]))
            # x_k2 = x_current + 0.5 * self.dt * k2
            # k3 = self.model.dynamic(x=x_k2, u=u_current, curvature=s_2_kappa(x_k2[0]))
            # x_k3 = x_current + self.dt * k3
            # k4 = self.model.dynamic(x=x_k3, u=u_current, curvature=s_2_kappa(x_k3[0]))
            # x_next = x_current + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x_next = k1 *self.dt
            g.append(X[(i+1)*self.nx: (i+2)*self.nx] - x_next)
            J += self.dt

        # terminal state constrain
        g.append(X[self.N*self.nx: ] - xf)

        # *************************** End N equal constrain ************************************
        lbg = [0]*(len(g)*self.nx)
        ubg = [0]*(len(g)*self.nx)

        # *************************** Unequal constrain ****************************************
        umax, umin = [self.model.velocity_limit, self.model.steering_limit], [-self.model.velocity_limit, -self.model.steering_limit]

        g.append(U-ca.repmat(umax, self.N))
        g.append(ca.repmat(umin, self.N) - U)

        # *************************** End 3 Unequal constrain ****************************************
        lbg.extend([-ca.inf]*2*self.nu*self.N)
        ubg.extend([0]*2*self.nu*self.N)

        g.append(-T_var)
        lbg.extend([-ca.inf])
        ubg.extend([0])

        nlp = {'x': ca.vertcat(X, U, T_var), 'f': J, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
        solver = ca.nlpsol('TOpt_solver', 'ipopt', nlp, opts)

        # x0 = ca.MX([0, 0, 0, 0, 0])
        T_guess = 4
        # Solve optimization problem
        # print(X.shape[0])
        print(len(lbg))
        res = solver(x0=ca.vertcat(ca.repmat(0, self.nx*(self.N+1), 1), ca.repmat(0, self.nu*self.N,1),T_guess), lbx=-ca.inf, ubx=ca.inf, lbg=lbg, ubg=ubg)

        # Extract optimal control policy
        U_opt = res['x'][self.nx * (self.N + 1):-1].full().reshape(self.nu, self.N)
        X_terminal= res['x'][:self.nx * (self.N + 1)].full().reshape(self.nx, self.N+1)
        T_opt = res['x'][-1]

        print("Optimal control sequence:")
        print(U_opt)
        print(T_opt)


if __name__ == "__main__":
    # my_traj_A_star = ASTAR_Traj()
    # my_traj_A_star.process()
    time_opt = TIME_OPT_Traj()
    time_opt.process_single_path()
