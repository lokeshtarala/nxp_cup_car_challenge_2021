from cv_bridge.core import CvBridge
import rclpy
from rclpy.node import Node
import copy
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64
from nxp_cup_interfaces.msg import PixyVector
from time import sleep
from datetime import datetime
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import cv2
import sensor_msgs.msg
import reset
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point

class LineFollow(Node):

    def __init__(self):
        super().__init__('aim_line_follow')

        #image_variables
        pyramid_down_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Number of times to pyramid image down.')
        
        self.declare_parameter("pyramid_down", 2, 
            pyramid_down_descriptor)
        

        #variables
        self.speed_vector = Vector3()
        self.steer_vector = Vector3()
        self.cmd_vel_car = Twist()
        self.position = Point()
        self.axes = [0,0,0,0,0,0,0] 
        self.speed =0.0
        self.steer = 0.0
        self.pyrDown = self.get_parameter("pyramid_down").value
        self.bridge = CvBridge()
        self.cameraImageTopic = '/trackImage0/image_raw'
       

        self.camera_vector_topic = '/cupcar0/PixyVector'

        # Subscribers
        self.pixy_subscriber = self.create_subscription(
            PixyVector,
            self.camera_vector_topic,
            self.listener_callback,
            10)
        
        self.joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10)

        self.imageSub = self.create_subscription(sensor_msgs.msg.Image, 
                    self.cameraImageTopic, 
                    self.pixyImageCallback, 
                    qos_profile_sensor_data)
                #/trackImage0/image_raw
            
        self.pose_subscriber = self.create_subscription(
            Odometry,
            '/cupcar0/odom',
            self.odom_callback,
            10)

                

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cupcar0/cmd_vel', 10)

        



        # Timer setup
        # timer_period = 0.5 #seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0

    
    def odom_callback(self,msg):
        self.position = msg.pose.pose.position
        # print(f"present position: {self.position}")


    # def timer_callback(self):
    #     #TODO
    def joy_callback(self,msg):
        self.axes = msg.axes
        self.speed = self.axes[1]
        self.speed = (self.speed )*1.5  
        self.steer = self.axes[3]
        self.steer = (self.steer + 1.0)*0.65 - 0.65
        # print(self.steer)
        # print(self.speed)
        

    def pixyImageCallback(self,msg):
         # Scene from subscription callback
        scene = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        #deep copy and pyramid down image to reduce resolution
        scenePyr = copy.deepcopy(scene)
        if self.pyrDown > 0:
            for i in range(self.pyrDown):
                scenePyr = cv2.pyrDown(scenePyr)
        sceneDetect = copy.deepcopy(scenePyr)
        cv2.imshow(sceneDetect)
        # print(sceneDetect)
        # print(np.shape(sceneDetect))



    def listener_callback(self, msg):
        #TODO
        

        self.speed_vector.x = self.speed
        self.steer_vector.z = self.steer
        self.cmd_vel_car.linear = self.speed_vector
        self.cmd_vel_car.angular = self.steer_vector

        self.cmd_vel_publisher.publish(self.cmd_vel_car)

def main(args=None):
    
    rclpy.init(args=args)

    line_follow = LineFollow()
    rclpy.spin(line_follow)

    line_follow.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()