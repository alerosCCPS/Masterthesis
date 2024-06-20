import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # 导入你需要使用的消息类型
from src.ackermann_drive_msg.msg import AckermannDrive
from src.ackermann_drive_msg.msg import AckermannDriveStamped

class MessNode(Node):

    def __init__(self):
        super.__init__("mess_node")
        self.subscription = rclpy.create_subscription(

        )
def main():

    rclpy.init()

    # 创建一个 ROS 2 节点
    node = rclpy.create_node('python_node')

    # 输出当前运行的主题列表
    topic_names = node.get_topic_names_and_types()
    print("Current topics:")
    for topic_name, _ in topic_names:
        print(topic_name)

    # 创建一个订阅者，订阅 '/robot0/pose' 主题
    subscription = node.create_subscription(
        String,  # 这里假设要订阅的消息类型是 String，根据实际情况修改
        '/robot0/pose',
        callback,
        10  # 缓冲区大小
    )
    subscription  # 避免未使用的警告

    # 进入 ROS 2 主循环
    rclpy.spin(node)

    # 关闭节点
    node.destroy_node()
    rclpy.shutdown()

def callback(msg):
    # 订阅回调函数，处理接收到的消息
    print(f"Received: {msg.data}")

if __name__ == '__main__':
    print("check point")
    main()