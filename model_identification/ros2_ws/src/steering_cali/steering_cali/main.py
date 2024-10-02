import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Header, Bool, Float32
import os
# import matplotlib.pyplot as plt
import pandas as pd

Script_Root = os.path.abspath(os.path.dirname(__file__))

class CommPub(Node):

    def __init__(self, name):

        super().__init__(name)
        self.data_path = os.path.join(Script_Root, 'DATA')
        self.get_logger().info("create CommPub node !")

        self.comm_pub = self.create_publisher(
            AckermannDriveStamped,
            '/hamster2/command',
            1
        )
        self.yaw_sub = self.create_subscription(
            Imu,
            '/hamster2/imu',
            self.listener_callback,
            5
        )
        self.yaw_sub

        self.velocity = 0.2  # m/s
        self.delta = 5.0 # degree
        self.comm_frequency = 60  # Hz
        self.start_time = self.get_clock().now()
        self.period = 30 # s
        self.timer_ack = self.create_timer(1 / self.comm_frequency, self.callback)
        self.velocity_list = [
            # 0.2,
            0.3,
            # 0.4,
            # 0.5,
            # 0.6
        ]
        self.delta_list = list(map(float, [i for i in range(5,30,5)]))
        self.v_counter = 0
        self.delta_counter = 0
        self.columns = ["v"] + self.delta_list
        self.rx, self.ry, self.rz = [], [], []
        self.data = []

    # def update(self):
    #     now = self.get_clock().now()
    #     elapsed_time = ( now - self.start_time).nanoseconds / 1e9
    #     if elapsed_time > 8:
    #         self.start_time = now
    #         self.delta_counter += 1
    #         if self.delta_counter >= len(self.delta_list):
    #             self.save_data()
    #             self.v_counter +=1
    #             self.delta_counter = 0
    #     if self.v_counter >= len(self.velocity_list) and self.delta_counter >= len(self.delta_list):
    #
    #         self.destroy_node()
    #     self.velocity = self.velocity_list[self.v_counter]
    #     self.delta = self.delta_list[self.delta_counter]
    def update(self):
        now = self.get_clock().now()
        elapsed_time = ( now - self.start_time).nanoseconds / 1e9
        if elapsed_time > 8:
            self.save_data()
            self.destroy_node()
        self.velocity = 0.6
        self.delta = 25.0


    def callback(self):
        self.update()
        self.comm_pub.publish(self.create_ack_command())

    def create_ack_command(self):
        drive = AckermannDrive()
        drive.steering_angle = self.delta
        drive.steering_angle_velocity = 0.0
        drive.speed = self.velocity
        drive.acceleration = 0.0
        drive.jerk = 0.0

        msg_stamped = AckermannDriveStamped()
        msg_stamped.header = Header()
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        msg_stamped.drive = drive
        return msg_stamped

    def listener_callback(self, msg):
        x,y,z = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        # r_x, r_y, r_z = self.velocity / x if abs(x)>1e-4 else 0
        r_y = self.velocity / abs(y) if abs(y)>1e-4 else 0
        r_y = 0 if r_y>2 else r_y
        # r_z =self.velocity / abs(z) if abs(z) > 1e-4 else 0
        # self.get_logger().info(f"r_x: {r_x}, r_y: {r_y}, r_z: {r_z}")
        self.get_logger().info(f" r_y: {r_y}")
        # self.rx.append(r_x)
        self.ry.append(r_y)
        # self.rz.append(r_z)

    def save_data(self):
        # rx = self.rx[::10]
        ry = [i for i in self.ry if i != 0]
        ry = ry[10:-10:10]
        av = sum(ry)/len(ry)
        self.get_logger().info(f"aver = {av}")
        ry.append(av)
        # rz = self.rz[::10]
        # pd.DataFrame(rx).to_csv(os.path.join(self.data_path, 'rx.csv'),index=False)
        file_name = f"velocity0_{str(self.velocity).split('.')[-1]}_delta_f{int(self.delta)}.csv"
        pd.DataFrame(ry).to_csv(os.path.join(self.data_path, file_name), index=False)
        # pd.DataFrame(rz).to_csv(os.path.join(self.data_path, 'rz.csv'),index=False)
        self.get_logger().info(f"saving {file_name}")
        self.ry = []

def start_comm_pub(args=None):

    # init the client lib
    rclpy.init(args=args)

    comm_pub_handle = CommPub(name="comm_pub_node")
    comm_pub_handle.get_logger().info("create handle of comm_pub_handle")
    try:
        rclpy.spin(node=comm_pub_handle)
    except KeyboardInterrupt:
        pass
    finally:

        comm_pub_handle.destroy_node()
        rclpy.shutdown()
