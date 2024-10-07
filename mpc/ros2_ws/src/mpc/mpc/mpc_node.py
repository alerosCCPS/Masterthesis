import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from hamster_interfaces.msg import TrackingState
from std_msgs.msg import Header, Bool, Float32
import casadi as ca
import numpy as np
import pandas as pd
# from scipy.interpolate import splev, splrep
import time
# from visualization import SimPlotter, ResultePlotter
# from ekf import EKF
import types
import json
import os
import math


Script_Root = os.path.abspath(os.path.dirname(__file__))
deg2R = lambda x: ca.pi*(x/180)
r2Deg = lambda x: 180*x/np.pi

def load_path(path_file_name):
    df = pd.read_csv(os.path.join(Script_Root,"DATA", path_file_name, "path.csv"))
    s = df["s"].values
    curvature = df['curvature'].values
    interpolator = ca.interpolant("kappa_at_s", "bspline", [s], curvature)
    return interpolator, len(s), s[-1], curvature[0]


def get_hamster_model(path_file_name='test_traj_normal'):
    _, _, path_length, kappa0 = load_path(path_file_name)
    model = types.SimpleNamespace()
    # model.mass = 1.5  # kg
    model.length_front = 0.125  # m
    model.length_rear = 0.125  # m
    model.L = (model.length_rear + model.length_front)
    model.steer_k1 = 2.25
    model.steer_k2 = 4
    model.T = 0.00984
    model.init_kappa = kappa0

    def dynamic_f(x, u, kappa):
        s, n, alpha, v = x[0], x[1], x[2], x[3]
        v_command, delta = u[0], u[1]

        delta = model.steer_k1 * model.L * ca.tanh(model.steer_k2 * delta)

        beta = model.length_rear *delta/model.L
        s_dot = v*ca.cos(alpha+beta)/(1-n*kappa)
        n_dot = v*ca.sin(alpha+beta)
        alpha_dot = v*ca.sin(beta)/model.length_rear - kappa*s_dot
        v_dot = (v_command-v)/model.T
        return ca.vertcat(s_dot, n_dot, alpha_dot, v_dot)

    # state vector
    s = ca.MX.sym("s")
    n = ca.MX.sym("n")
    alpha = ca.MX.sym("alpha")
    v = ca.MX.sym("v")
    x = ca.vertcat(s, n, alpha, v)
    kappa = ca.MX.sym("kappa")

    # control vector
    v_comm = ca.MX.sym("v_comm")
    delta = ca.MX.sym("delta")
    u = ca.vertcat(v_comm, delta)

    rhs = dynamic_f(x, u, kappa)

    model.dynamic = ca.Function("dynamic", [x,u,kappa], [rhs])

    constraints = types.SimpleNamespace()
    constraints.s_limit = path_length  # total curve length
    with open(os.path.join(Script_Root, "setup.json"))as f:
        data = json.load(f)
    constraints.n_limit = data["constraints"]["n"]
    constraints.alpha_limit = ca.pi*data["constraints"]["alpha"]
    constraints.v_limit = data["max_vel"]
    constraints.v_comm_limit = data["max_vel"]
    constraints.targ_vel = data["targ_vel"]
    constraints.delta_limit = deg2R(data["constraints"]["delta"])

    print("set constraints:")
    print(constraints)

    return model, constraints

#
# class MPC:
#
#     def __init__(self, path_name='test_traj_normal'):
#         self.simple_mode = False
#         self.save_root = os.path.join(Script_Root, "DATA", path_name)
#         self.interpolator,_,self.s_max, _ = load_path(path_name)
#         # self.EKF = EKF(path_name)
#         self.hamster, self.constraints = get_hamster_model(path_name)
#
#         self.sample_time = 0.1
#         self.N = 20
#         self.Q = np.diag([0, 0, 0, 1e-1])
#         self.QN = np.diag([0, 0, 0, 1e-1])
#         self.R = np.diag([1e-8, 1e-8])
#         self.load_setup()
#
#         self.nu = 2
#         self.nx = 4
#         self.X = ca.MX.sym("X", self.nx, self.N + 1)
#         self.U = ca.MX.sym("U", self.nu, self.N)
#         self.P = ca.MX.sym("P", 2 * self.nx)
#
#         self.J = 0
#         self.g = []
#         self.lbg = []
#         self.ubg = []
#         self.lbx = []
#         self.ubx = []
#         self.U_opt = []
#         self.X_opt = []
#         self.x0_ = None
#         self.solver = None
#         self.OCP_results_X = []  #
#         self.OCP_results_U = []
#         self.Solverstats = []  # store status of solver for each OCP
#         self.feasible_iterations = []
#         self.setup_nlp()
#
#     def setup_nlp(self):
#         # initial constrain
#         self.g.append(self.X[:, 0] - self.P[0:self.nx])
#         if self.simple_mode:
#             s = ca.if_else(self.X[:, 0][0]>self.s_max, self.X[:, 0][0]-self.s_max, self.X[:, 0][0])
#             kappa = self.interpolator(s)
#
#         con_ref = np.array([0, 0])
#         # system dynamic constrain
#         for i in range(self.N):
#
#             if not self.simple_mode:
#                 s = ca.if_else(self.X[:, i][0] > self.s_max, self.X[:, i][0] - self.s_max, self.X[:, i][0])
#                 kappa = self.interpolator(s)
#             state, con = self.X[:, i], self.U[:, i]
#             x_error = state - self.P[self.nx:]
#
#             con_diff = con - con_ref
#             con_ref = con
#             self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error)\
#                       +ca.mtimes(ca.mtimes(con_diff.T, self.R), con_diff)
#             state_dot = self.hamster.dynamic(state, con, kappa)
#             state_next = state + self.sample_time * state_dot
#             self.g.append(state_next - self.X[:, i + 1])
#         x_error = self.X[:,-1] - self.P[self.nx:]
#         self.J += ca.mtimes(ca.mtimes(x_error.T, self.QN), x_error)
#
#         self.g = ca.vertcat(*self.g)
#         self.lbg = [0] * self.g.size()[0]
#         self.ubg = [0] * self.g.size()[0]
#         self.lbx = [-ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)
#         self.ubx = [ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)
#
#         # constrain s
#         self.lbx[0:self.nx * (self.N + 1):4] = [0] * len(self.lbx[0:self.nx * (self.N + 1):4])
#         # self.ubx[0:self.nx * (self.N + 1):4] = [self.constraints.s_limit] * len(self.ubx[0:self.nx * (self.N + 1):4])
#
#         # constrain n
#         self.lbx[1:self.nx * (self.N + 1):4] = [-self.constraints.n_limit] * len(self.lbx[1:self.nx * (self.N + 1):4])
#         self.ubx[1:self.nx * (self.N + 1):4] = [self.constraints.n_limit] * len(self.ubx[1:self.nx * (self.N + 1):4])
#
#         # constrain alpha
#         self.lbx[2: self.nx * (self.N + 1):4] = [-self.constraints.alpha_limit] * len(self.lbx[2: self.nx * (self.N + 1):4])
#         self.ubx[2: self.nx * (self.N + 1):4] = [self.constraints.alpha_limit] * len(self.ubx[2: self.nx * (self.N + 1):4])
#
#         # constrain v
#         self.lbx[3: self.nx * (self.N + 1): 4] = [0] * len(self.lbx[3: self.nx * (self.N + 1): 4])
#         self.ubx[3: self.nx * (self.N + 1): 4] = [self.constraints.v_limit] * len(self.ubx[3: self.nx * (self.N + 1): 4])
#
#         # constrain v_comm
#         self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
#         self.ubx[self.nx * (self.N + 1):: 2] = [self.constraints.v_comm_limit] * len(self.ubx[self.nx * (self.N + 1):: 2])
#
#         # constrain delta
#         self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constraints.delta_limit] * len(self.lbx[self.nx * (self.N + 1) + 1:: 2])
#         self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constraints.delta_limit] * len(self.ubx[self.nx * (self.N + 1) + 1:: 2])
#
#         opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-6}
#         nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J,
#                'g': self.g, 'p': self.P}
#         self.solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
#
#         print("set up nlp solver")
#
#     def predict(self, x=[0, 0, 0, 0]):
#         if self.x0_ is None:
#             self.x0_ = ca.repmat(0, self.nx * (self.N + 1) + self.nu * self.N, 1)
#         p = x + [0, 0, 0, self.constraints.v_limit]
#         res = self.solver(x0=self.x0_, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
#         self.x0_ = res["x"]
#         con = ca.reshape(self.x0_[self.nx * (self.N + 1):], self.nu, self.N).full()
#         status = self.solver.stats()  # solver status
#         if status['return_status'] == 'Infeasible_Problem_Detected':
#             print("OCP seems to be infeasible, please check carefully")
#         else:
#             print("OPC is feasible")
#         return con[:, 0]
#
#     def load_setup(self):
#         with open(os.path.join(Script_Root, "setup.json"), "r")as f:
#             data = json.load(f)
#             self.Q = np.diag(data["Q"])
#             self.QN = np.diag(data["QN"])
#             self.R = np.diag(data["R"])
#             self.N = data["N"]
#             self.sample_time = data["sample_time"]
#             self.simple_mode = data["simple_mode"]
#             print("load set up data: ")
#             print(f"Q: {self.Q}")
#             print(f"QN: {self.QN}")
#             print(f"R: {self.R}")
#             print(f"sample time step: {self.sample_time}")
#             print(f"prediction horizon: {self.N}")
#             print(f"simple mode: {self.simple_mode}")

class MPC_Adapted:

    def __init__(self, path_name='test_traj'):
        self.threshold_delta = 25
        self.No_track = False
        self.kappa_ref = 0
        self.x_init = [0, 0, 0, 0]
        self.path_name = path_name
        self.interpolator, _, self.s_max,_ = load_path(self.path_name)
        self.save_root = os.path.join(Script_Root, "DATA", path_name)
        # self.EKF = EKF(path_name)
        self.hamster, self.constraints = get_hamster_model(path_name)
        self.controller_freq = 10  # second, Hz
        self.sample_time = 0.1
        self.N = 10
        self.nu = 2
        self.nx = 4
        self.simple_mode = False

        self.Q = np.diag([0, 1, 1, 1e-1])
        self.QN = np.diag([0, 1, 1, 1e-1])
        self.R = np.diag([1, 1, 1e-1, 1e-1])
        self.load_setup()
        self.X = ca.MX.sym("X", self.nx, self.N + 1)
        self.U = ca.MX.sym("U", self.nu, self.N)
        self.P = ca.MX.sym("P", 2 * self.nx)

        self.J = 0
        self.g = []  # currently forced to zero later (system dynamic constraints; multiple shooting)
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.U_opt = []
        self.X_opt = []
        # to store more of what is happening; this might seem unnecessary, but for me it was almost always useful to undertsnad what is happening and to make sure my code/ocp makes sense
        self.OCP_results_X = []  #
        self.OCP_results_U = []
        self.Solverstats = []  # store status of solver for each OCP
        self.feasible_iterations = []  # to check if the ocp was always feasible in closed loop

        self.x0_ = None
        self.solver = None
        self.setup_nlp()

    def setup_nlp(self):
        # initial constrain
        self.g.append(self.X[:, 0] - self.P[0:self.nx])

        if self.simple_mode:
            s = ca.if_else(self.X[:, 0][0] > self.s_max, self.X[:, 0][0] - self.s_max, self.X[:, 0][0])
            kappa = self.interpolator(s)

        con_pre = np.array([0, 0])
        # system dynamic constrain
        for i in range(self.N):
            state, con = self.X[:, i], self.U[:, i]

            if not self.simple_mode:
                s = ca.if_else(self.X[:, i][0] > self.s_max, self.X[:, i][0] - self.s_max, self.X[:, i][0])
                kappa = self.interpolator(s)
            L =self.hamster.length_front + self.hamster.length_rear
            l_r = self.hamster.length_rear

            alpha_ref = -ca.asin(kappa * l_r)
            # delta_ref = L/l_r * ca.asin(kappa*l_r)
            # delta_ref = L/l_r * ca.asin((kappa-self.hamster.steer_c)*l_r/self.hamster.steer_k)

            delta_ref = ca.arctanh(kappa/self.hamster.steer_k1)/self.hamster.steer_k2
            delta_ref = ca.if_else(delta_ref > deg2R(self.threshold_delta), deg2R(self.threshold_delta), delta_ref )
            delta_ref = ca.if_else(delta_ref < -deg2R(self.threshold_delta), -deg2R(self.threshold_delta), delta_ref)
            con_error = con - ca.vertcat(self.P[-1], delta_ref)
            con_diff = con - con_pre
            con_pre = con
            con_join = ca.vertcat(con_error, con_diff)
            x_error = state - ca.vertcat(0,0,alpha_ref,self.P[-1])
            self.J += ca.mtimes(ca.mtimes(x_error.T, self.Q), x_error) \
                      + ca.mtimes(ca.mtimes(con_join.T, self.R), con_join)
            state_dot = self.hamster.dynamic(state, con, kappa)
            state_next = state + self.sample_time * state_dot
            self.g.append(state_next - self.X[:, i + 1])  # multiple shooting constraint
        if not self.simple_mode:
            s = ca.if_else(self.X[:, -1][0] > self.s_max, self.X[:, -1][0] - self.s_max, self.X[:, -1][0])
            kappa = self.interpolator(s)
        alpha_ref = -ca.asin(kappa * self.hamster.length_rear)
        x_error = self.X[:, -1] - ca.vertcat(0,0,alpha_ref,self.P[-1])
        self.J += ca.mtimes(ca.mtimes(x_error.T, self.QN), x_error)

        self.g = ca.vertcat(*self.g)
        self.lbg = [0] * self.g.size()[0]
        self.ubg = [0] * self.g.size()[0]
        self.lbx = [-ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)
        self.ubx = [ca.inf] * (self.nx * (self.N + 1) + self.nu * self.N)

        # constrain s
        self.lbx[0:self.nx * (self.N + 1):4] = [0] * len(self.lbx[0:self.nx * (self.N + 1):4])
        # self.ubx[0:self.nx * (self.N + 1):4] = [self.constraints.s_limit] * len(self.ubx[0:self.nx * (self.N + 1):4])

        # constrain n
        self.lbx[1:self.nx * (self.N + 1):4] = [-self.constraints.n_limit] * len(self.lbx[1:self.nx * (self.N + 1):4])
        self.ubx[1:self.nx * (self.N + 1):4] = [self.constraints.n_limit] * len(self.ubx[1:self.nx * (self.N + 1):4])

        # constrain alpha
        self.lbx[2: self.nx * (self.N + 1):4] = [-self.constraints.alpha_limit] * len(
            self.lbx[2: self.nx * (self.N + 1):4])
        self.ubx[2: self.nx * (self.N + 1):4] = [self.constraints.alpha_limit] * len(
            self.ubx[2: self.nx * (self.N + 1):4])

        # constrain v
        self.lbx[3: self.nx * (self.N + 1): 4] = [0] * len(self.lbx[3: self.nx * (self.N + 1): 4])
        self.ubx[3: self.nx * (self.N + 1): 4] = [self.constraints.v_limit] * len(
            self.ubx[3: self.nx * (self.N + 1): 4])

        # constrain v_comm
        self.lbx[self.nx * (self.N + 1):: 2] = [0] * len(self.lbx[self.nx * (self.N + 1):: 2])
        self.ubx[self.nx * (self.N + 1):: 2] = [self.constraints.v_comm_limit] * len(
            self.ubx[self.nx * (self.N + 1):: 2])

        # constrain delta
        self.lbx[self.nx * (self.N + 1) + 1:: 2] = [-self.constraints.delta_limit] * len(
            self.lbx[self.nx * (self.N + 1) + 1:: 2])
        self.ubx[self.nx * (self.N + 1) + 1:: 2] = [self.constraints.delta_limit] * len(
            self.ubx[self.nx * (self.N + 1) + 1:: 2])

        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.tol': 1e-6}
        nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J,
               'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)

        print("set up nlp solver")

    def predict(self, x=[0, 0, 0, 0]):
        if self.x0_ is None:
            x0x = ca.repmat([0,0,0,0], (self.N + 1))
            x0u = ca.repmat([self.constraints.targ_vel, self.hamster.length_rear * self.hamster.init_kappa], self.N)
            self.x0_ = ca.vertcat(x0x, x0u)
            # self.x0_ = ca.repmat(0, self.nx * (self.N + 1) + self.nu * self.N, 1)
        p = x + [0, 0, 0, self.constraints.targ_vel]

        res = self.solver(x0=self.x0_, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        self.x0_ = res["x"]
        con = ca.reshape(self.x0_[self.nx * (self.N + 1):], self.nu, self.N).full()  # planned control input (due to nlp formulation they are stored at the end of the solution vector)

        # store all OCP results
        status = self.solver.stats()  # solver status
        if status['return_status'] == 'Infeasible_Problem_Detected':
            print("OCP seems to be infeasible, please check carefully")
        else:
            # print("OPC is feasible")
            pass

        # U_OCP = con  # planned input sequence
        # X_OCP = ca.reshape(self.x0_[0:self.nx * (self.N + 1)], self.nx, self.N + 1).full()  # predicted state sequence

        u_star = con[:,0]
        u_star[0] = 0.2 if u_star[0]<0.2 else u_star[0]
        return u_star

    def load_setup(self):
        with open(os.path.join(Script_Root, "setup.json"), "r")as f:
            data = json.load(f)
            self.Q = np.diag(data["Q"])
            self.QN = np.diag(data["QN"])
            self.R = np.diag(data["R"])
            self.N = data["N"]
            self.sample_time = data["sample_time"]
            self.simple_mode = data["simple_mode"]
            print("load set up data: ")
            print(f"Q: {self.Q}")
            print(f"QN: {self.QN}")
            print(f"R: {self.R}")
            print(f"sample time step: {self.sample_time}")
            print(f"prediction horizon: {self.N}")
            print(f"simple mode: {self.simple_mode}")
class Controller(Node):

    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("create Controller Node !")
        self.data_root = os.path.join(Script_Root, "DATA",name)
        # self.mpc = MPC(name)
        self.mpc = MPC_Adapted(name)
        self.controller_frequency = 10  # Hz
        self.constant_velocity = False
        with open(os.path.join(Script_Root, "setup.json"))as f:
            data = json.load(f)
            self.DEBUG = data["DEBUG"]
            self.controller_frequency = data["frequency"]  # Hz
            self.constant_velocity = data["constant_velocity"]
            self.target_v = data["targ_vel"]
        self.get_logger().info(f"DEBUG MODE {self.DEBUG}")
        self.get_logger().info(f"controller frequency: {self.controller_frequency} Hz")

        self.state = [0,0,0,0]  # s, n, alpha, v
        self.timer_controller = self.create_timer(1 / self.controller_frequency, self.controller_callback)
        self.msg_stamped = AckermannDriveStamped()
        self.init_msg()
        self.his = []

        self.comm_pub = self.create_publisher(
            AckermannDriveStamped,
            '/hamster2/command',
            1
        )

        self.vel_listener = self.create_subscription(
            Float32,
            '/hamster2/velocity',
            self.vel_callback,
            5
        )

        self.track_listener = self.create_subscription(
            TrackingState,
            "/hamster2/tracking_state",
            self.track_callback,
            5
        )
        self.vel_listener
        self.track_listener

    def vel_callback(self, msg):
        self.state[-1] = msg.data
    def track_callback(self, msg):
        self.state[0] = msg.path_progress
        self.state[1] = msg.lat_dev
        self.state[2] = msg.head_dev

    def controller_callback(self):
        self.msg_stamped.header.stamp = self.get_clock().now().to_msg()
        start = time.time()
        u = self.mpc.predict(x=self.state)
        duration = time.time()-start

        self.msg_stamped.drive.speed = self.target_v if self.constant_velocity else u[0]
        self.msg_stamped.drive.steering_angle = r2Deg(u[1])
        self.comm_pub.publish(self.msg_stamped)
        self.his.append(self.state + list(u) + [duration])
        if self.DEBUG:
            self.get_logger().info(f"tracking state: {self.state}")
            self.get_logger().info(f"control: {u}")

    def init_msg(self):
        self.msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.msg_stamped.drive.steering_angle = 0.0
        self.msg_stamped.drive.steering_angle_velocity = 0.0
        self.msg_stamped.drive.speed = 0.0
        self.msg_stamped.drive.acceleration = 0.0
        self.msg_stamped.drive.jerk = 0.0

    def save_data(self):
        heads = ['s', 'n', 'alpha', 'v', 'v_comm', 'delta', 'time']
        data = np.vstack(self.his)
        df = pd.DataFrame(data, columns=heads)
        df.to_csv(os.path.join(self.data_root,"real_results.csv"), index=False)

def start_controller(args=None):

    with open(os.path.join(Script_Root,"setup.json"))as f:
        data = json.load(f)
        dir = data["direction"]
        sim = data["simple_mode"]
        constant = data["constant_velocity"]
    if not dir:
        case_name = "test_traj_reverse"
    else:
        if constant:
            case_name = "test_traj_mpc_simple_constant" if sim else "test_traj_mpc_constant"
        else:
            # case_name = "val_traj_mpc_simple" if sim else "val_traj_mpc"
            # case_name = "test_traj_mpc_simple" if sim else "test_traj_mpc"
            # case_name = "val_traj_mpc_adapted_simple" if sim else "val_traj_mpc_adapted"
            case_name = "test_traj_mpc_adapted_simple" if sim else "test_traj_mpc_adapted"
    rclpy.init(args=args)

    controller_handle = Controller(name=case_name)
    controller_handle.get_logger().info("create handle of comm_pub_handle")
    try:
        rclpy.spin(node=controller_handle)
    except KeyboardInterrupt:
        pass
    finally:
        controller_handle.save_data()
        controller_handle.destroy_node()
        rclpy.shutdown()