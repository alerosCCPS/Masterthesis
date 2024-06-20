import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from std_msgs.msg import Header

class CommPub(Node):

    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("create CommPub node !")
        self.comm_pub = self.create_publisher(
            AckermannDriveStamped,
            '/command_info',
            1
        )
        timer_period = 1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.comm_pub.publish(self.creat_command())

    def creat_command(self):
        drive = AckermannDrive()
        drive.steering_angle = 0.0
        drive.steering_angle_velocity = 0.0
        drive.speed = 1.0
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
            '/command_info',
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