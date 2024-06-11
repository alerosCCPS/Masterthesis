import types

from hamster_dynamic import Hasmster, get_casadi_model
from scipy.interpolate import interp1d
import casadi as ca
import pandas as pd
import numpy as np
from typing import Optional
import time
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


class OCP_Traj:

    def __init__(self):
        self.model = get_casadi_model()
        self.dt: Optional[float] = 0.02  # Time step (second)
        self.N: Optional[int] = 250  # Prediction horizon
        self.nx: Optional[int] = 4
        self.nu: Optional[int] = 2
        self.path_root = os.path.join(Script_Root, 'DATA', 'path')
        self.path_list = os.listdir(self.path_root)
        self.path_of_path = []
        self.U_opt: Optional[np.ndarray] = None
        self.X_opt: Optional[np.ndarray] = None

    def create_trajectory(self):
        trajectory = types.SimpleNamespace()
        trajectory.U_opt = self.U_opt
        trajectory.X_opt = self.X_opt
        trajectory.path_of_path = self.path_of_path
        trajectory.steps = self.N
        return trajectory

    def process_single_path(self, case_name='circle'):

        self.path_of_path = os.path.join(self.path_root, case_name, 'path.csv')
        # load arc length and curvature
        df = pd.read_csv(self.path_of_path)
        s, kappa = list(df["s"].values), list(df["curvature"].values)

        # define state and control vector
        U = ca.MX.sym('U', self.nu, self.N)  # Input sequence
        X = ca.MX.zeros(self.nx, self.N +1)
        xf = ca.MX([s[-1], 0, 0, 0])
        x0 = ca.MX([0,0,0,0])
        X[:,0] = x0

        g = []
        lbg = []
        ubg = []

        # self.dt = 5/self.N

        for i in range(self.N):
            x_current, u_current = X[:, i], U[:, i]
            curvature = 0.5
            dx = self.model.dynamic(x=x_current, u=u_current, curvature=curvature)
            x_next = x_current + dx * self.dt
            X[:, i+1] = x_next

            # constrain positive s
            g.append(-X[0, i+1])
            lbg.append(-ca.inf)
            ubg.append(0)

            # constrain n
            # g.append(X[1, i+1] - 1/curvature)
            g.append(X[1, i + 1] - 0.01)
            g.append(-0.01 - X[1, i + 1])
            lbg.extend([-ca.inf]*2)
            ubg.extend([0]*2)

            # constrain alpha
            rotation_tol = np.pi*0.25
            g.append(X[2, i+1] - rotation_tol)
            g.append(-rotation_tol - X[2, i+1])
            lbg.extend([-ca.inf]*2)
            ubg.extend([0]*2)

            # constrain velocity limit
            g.append(X[3, i+1] - self.model.velocity_limit)
            g.append(-self.model.velocity_limit - X[3, i+1])
            lbg.extend([-ca.inf]*2)
            ubg.extend([0]*2)

            # J += ca.norm_2(x_next - xf)**2
        J = ca.norm_2(X[:, -1] - xf)**2
        # terminal state constrain
        # g.append(state_history[-1] - xf)
        # lbg.extend([0]*self.nx)
        # ubg.extend([0]*self.nx)

        umax, umin = [self.model.velocity_limit, self.model.steering_limit], [-self.model.velocity_limit, -self.model.steering_limit]

        # g.append(U-ca.repmat(umax, self.N))
        # g.append(ca.repmat(umin, self.N) - U)
        # lbg.extend([-ca.inf]*2*self.nu*self.N)
        # ubg.extend([0]*2*self.nu*self.N)
        lbx = umin * self.N
        ubx = umax * self.N

        nlp = {'x':  ca.reshape(U, -1, 1), 'f': J, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
        solver = ca.nlpsol('TOpt_solver', 'ipopt', nlp, opts)

        # Solve optimization problem
        start = time.time()
        res = solver(x0=ca.repmat(0, self.nu * self.N, 1), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        duration = time.time() - start
        print("Duration: ", duration)

        # constrain_value = res['g'].full()
        # print("terminal constrain value: ",constrain_value)

        # Extract optimal trajectory
        self.U_opt = ca.reshape(res['x'], self.nu, self.N).full()
        F = ca.Function('F', [U], [X])
        self.X_opt = F(self.U_opt).full()

        return self.create_trajectory()


if __name__ == "__main__":
    # my_traj_A_star = ASTAR_Traj()
    # my_traj_A_star.process()
    time_opt = OCP_Traj()
    time_opt.process_single_path()
