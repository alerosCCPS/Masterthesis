import numpy as np
import casadi as ca
import types

deg2R = lambda x: np.pi*(x/180)


class Hasmster:

    def __init__(self):
        self.steering_limit = deg2R(14)  # 14 degree
        self.velocity_limit = 1.2  # m/s
        self.mass = 1.5  # kg
        self.length_front = 0.125  # m
        self.length_rear = 0.125  # m
        self.T = 1
        # self.x = [0,0,0,0]  # s,n,alpha,v
        # self.x_dot = [0,0,0,0]

    def update(self,state,  u, curvature:float):
        """
        :param u: control: v_command, delta
        :param curvature:
        :return:
        """
        s, n, alpha, v = state
        v_command, delta = u
        beta = np.arctan2(self.length_rear*np.tan(delta), (self.length_rear + self.length_front))
        s_dot = v*np.cos(alpha+beta)/(1-n*curvature)
        n_dot = v*np.sin(alpha+beta)
        alpha_dot = v*np.sin(beta)/self.length_rear - curvature*s_dot
        v_dot = (v_command-v)/self.T

        return [s_dot, n_dot, alpha_dot, v_dot]


def get_casadi_model():
    model = types.SimpleNamespace()
    # constrains = types.SimpleNamespace()
    model.steering_limit = deg2R(14)  # 14 degree
    model.velocity_limit = 0.6  # m/s
    model.mass = 1.5  # kg
    model.length_front = 0.125  # m
    model.length_rear = 0.125  # m
    model.T = 0.00984

    def dynamic_f(x, u, curvature):
        s, n, alpha, v = x[0], x[1], x[2], x[3]
        v_command, delta = u[0], u[1]
        # beta = np.arctan2(model.length_rear * np.tan(delta), (model.length_rear + model.length_front))
        beta = 0.5*delta
        s_dot = v*np.cos(alpha+beta)/(1-n*curvature)
        n_dot = v*np.sin(alpha+beta)
        alpha_dot = v*np.sin(beta)/model.length_rear - curvature*s_dot
        v_dot = (v_command-v)/model.T
        return ca.vertcat(s_dot, n_dot, alpha_dot, v_dot)

    model.dynamic = dynamic_f

    # s = ca.MX.sym("s")
    # n = ca.MX.sym("n")
    # alpha = ca.MX.sym("alpha")
    # v = ca.MX.sym("v")
    # # t_var = ca.MX.sym("t_var")
    # x = ca.vertcat(s, n, alpha, v)
    # v_command, delta = ca.MX.sym("v_command"), ca.MX.sym("delta")
    # u = ca.vertcat(v_command, delta)
    # model.x = x
    # model.u = u

    return model
