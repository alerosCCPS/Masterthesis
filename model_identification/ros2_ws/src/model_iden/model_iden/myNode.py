import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from std_msgs.msg import Header, Bool, Float32
import os
import pandas as pd

Script_Root = os.path.abspath(os.path.dirname(__file__))

Target_velocity = 0.2  # m/s
Init_velocity = 0.6  # m/s


class CommPub(Node):

    def __init__(self, name):

        super().__init__(name)
        self.get_logger().info("create CommPub node !")

        self.comm_pub = self.create_publisher(
            AckermannDriveStamped,
            '/hamster2/command',
            1
        )

        # self.interlock_pub = self.create_publisher(
        #     Bool,
        #     '/hamster2/interlock',
        #     1
        # )

        self.target_velocity = Target_velocity  # m/s
        self.init_velocity = Init_velocity  # m/s
        self.init_time_period = 4  # s
        self.start_time = self.get_clock().now()
        self.comm_frequency = 10  # Hz
        self.timer_ack = self.create_timer(1 / self.comm_frequency, self.timer_ack_callback)
        # self.timer_interlock = self.create_timer(0.003, self.timer_interlock_callback)

    def timer_ack_callback(self):
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed_time < self.init_time_period:
            self.comm_pub.publish(self.create_ack_command(self.init_velocity))
        else:
            self.comm_pub.publish(self.create_ack_command(self.target_velocity))

    def timer_interlock_callback(self):
        self.interlock_pub.publish(self.create_interlock())

    def create_interlock(self):
        msg = Bool()
        msg.data = True
        return msg

    def create_ack_command(self, velocity):
        drive = AckermannDrive()
        drive.steering_angle = 0.0
        drive.steering_angle_velocity = 0.0
        drive.speed = velocity
        drive.acceleration = 0.0
        drive.jerk = 0.0

        msg_stamped = AckermannDriveStamped()
        msg_stamped.header = Header()
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        msg_stamped.drive = drive
        return msg_stamped


class CommSub(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("create CommSub node !")
        self.comm_sub = self.create_subscription(
            AckermannDriveStamped,
            '/hamster2/command',
            self.listener_callback,
            10
        )
        self.comm_sub

    def listener_callback(self, msg):
        self.get_logger().info('get command:')
        # 打印header信息
        self.get_logger().info('Header:')
        self.get_logger().info(f'  Frame ID: {msg.header.frame_id}')
        self.get_logger().info(f'  Stamp: {msg.header.stamp.sec} sec, {msg.header.stamp.nanosec} nanosec')

        # 打印drive信息
        self.get_logger().info('Drive:')
        self.get_logger().info(f'  Steering Angle: {msg.drive.steering_angle}')
        self.get_logger().info(f'  Steering Angle Velocity: {msg.drive.steering_angle_velocity}')
        self.get_logger().info(f'  Speed: {msg.drive.speed}')
        self.get_logger().info(f'  Acceleration: {msg.drive.acceleration}')
        self.get_logger().info(f'  Jerk: {msg.drive.jerk}')

class MessSub(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("create velocity listener !")

        self.listener = self.create_subscription(
            Float32,
            '/hamster2/velocity',
            self.listener_callback,
            10
        )
        self.listener
        self.save_root = os.path.join(Script_Root, 'DATA')
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        self.mess_data = {
            "velocity": [],
            "init_velocity": Init_velocity,
            "target_velocity": Target_velocity
        }

    def listener_callback(self, msg):
        self.get_logger().info(f'velocity: {msg.data}')
        self.mess_data["velocity"].append(msg.data)

    def save_data(self):
        df = pd.DataFrame(self.mess_data)
        df.to_csv(os.path.join(self.save_root, 'velocity.csv'), index=False)

class DataSender(Node):

    def __init__(self,name):
        super().__init__(name)
        self.data_pub = self.create_publisher(
            Float32,
            '/hamster2/velocity',
            1
        )
        self.v = 0.0
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = Float32()
        msg.data = self.v
        self.v += 1.0
        self.data_pub.publish(msg)

def start_comm_pub(args=None):

    # init the client lib
    rclpy.init(args=args)

    comm_pub_handle = CommPub(name="comm_pub_node")
    comm_pub_handle.get_logger().info("create handle of comm_pub_handle")
    rclpy.spin(node=comm_pub_handle)

    # shutdown the node
    rclpy.shutdown()

def start_comm_sub(args=None):

    # init the client lib
    rclpy.init(args=args)

    comm_sub_handle = CommSub(name="comm_sub_node")
    comm_sub_handle.get_logger().info("create handle of comm_pub_handle")
    rclpy.spin(node=comm_sub_handle)

    # shutdown the node
    rclpy.shutdown()

def start_mess_sub(args=None):
    rclpy.init(args=args)

    mess_sub_handle = MessSub(name="mess_sub_node")
    mess_sub_handle.get_logger().info("create handle mess_sub_handle")

    try:
        rclpy.spin(node=mess_sub_handle)
    except KeyboardInterrupt:
        pass
    finally:
        mess_sub_handle.save_data()
        mess_sub_handle.destroy_node()
        rclpy.shutdown()

def start_data_sender(args=None):
    rclpy.init(args=args)

    data_sender_handle = DataSender(name="data_sender_node")
    data_sender_handle.get_logger().info("create handle data_sender_handle")

    rclpy.spin(node=data_sender_handle)

    rclpy.shutdown()