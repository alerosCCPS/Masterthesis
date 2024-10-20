import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from hamster_interfaces.msg import TrackingState
from std_msgs.msg import Header, Bool, Float32
from gp_class import GP3D_LF
import torch
import numpy as np
import pandas as pd
import time
import json
import os

Script_Root = os.path.abspath(os.path.dirname(__file__))
deg2R = lambda x: np.pi*(x/180)
r2Deg = lambda x: 180*x/np.pi


class Controller_GP3D_LF(Node):

    def __init__(self, name):

        super().__init__(name)
        self.data_type = torch.float32
        self.get_logger().info("create Controller Node !")
        # self.data_root = os.path.join(Script_Root, "DATA",f"test_traj_3D_LF")
        self.data_root = os.path.join(Script_Root, "DATA", f"val_traj_3D_LF")
        self.mpc = GP3D_LF(self.data_root)
        with open(os.path.join(Script_Root, "setup.json"))as f:
            data = json.load(f)
            self.DEBUG = data["DEBUG"]
            self.target_v = data["target_v"]
            self.controller_frequency = data['frequency']

        self.get_logger().info(f"DEBUG MODE: {self.DEBUG}")
        self.get_logger().info(f"target_v: {self.target_v}")
        self.get_logger().info(f"controller frequency: {self.controller_frequency}")

        self.state = [0,0,0,0]  # s, n, alpha, v
        self.x = torch.tensor([[0, 0, 0, 0]]).to(self.data_type)  # kappa, look_fw_curvature, n, alpha,
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

        self.x[0, 0] = torch.tensor(msg.curvature)
        self.x[0, 1] = torch.tensor(msg.curvature_lf)
        self.x[0, 2] = torch.tensor(msg.lat_dev)
        self.x[0, 3] = torch.tensor(msg.head_dev)

    def controller_callback(self):
        self.msg_stamped.header.stamp = self.get_clock().now().to_msg()
        start = time.time()
        u = self.mpc.predict(x=self.x)
        duration = time.time() - start

        self.msg_stamped.drive.speed = self.target_v
        self.msg_stamped.drive.steering_angle = r2Deg(u)
        self.comm_pub.publish(self.msg_stamped)
        self.his.append([self.x[0,0].numpy().item()]+self.state + [self.target_v, u, duration])
        if self.DEBUG:
            self.get_logger().info(f"state: {self.state}")
            self.get_logger().info(f"control: {u}")

    def init_msg(self):
        self.msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.msg_stamped.drive.steering_angle = 0.0
        self.msg_stamped.drive.steering_angle_velocity = 0.0
        self.msg_stamped.drive.speed = 0.0
        self.msg_stamped.drive.acceleration = 0.0
        self.msg_stamped.drive.jerk = 0.0

    def save_data(self):
        heads = ['curvature', 's', 'n', 'alpha', 'v', 'v_comm', 'delta', 'time']
        data = np.vstack(self.his)
        df = pd.DataFrame(data, columns=heads)
        df.to_csv(os.path.join(self.data_root,"real_results.csv"), index=False)

def start_gp_3D_LF(args=None):

    rclpy.init(args=args)

    gp_3D_LF = Controller_GP3D_LF(name='gp_3D_LF')
    gp_3D_LF.get_logger().info("create handle of comm_pub_handle")
    try:
        rclpy.spin(node=gp_3D_LF)
    except KeyboardInterrupt:
        pass
    finally:
        gp_3D_LF.save_data()
        gp_3D_LF.destroy_node()
        rclpy.shutdown()