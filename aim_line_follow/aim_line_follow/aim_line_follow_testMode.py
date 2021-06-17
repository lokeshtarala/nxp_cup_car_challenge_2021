
from re import L
import rclpy
from .nxp_rl import *


from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from nxp_cup_interfaces.msg import PixyVector
from time import sleep
import time
from datetime import datetime
import numpy as np


from rclpy.node import Node
import os



import numpy as np

import cv2

import copy


from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point
from rclpy.task import Future
from std_srvs.srv import Empty
from cv_bridge.core import CvBridge


from . import nxp_rl





class LineFollow(Node):

    def __init__(self):
        super().__init__('aim_line_follow')


        start_delay_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Seconds to delay before starting.')

        camera_vector_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Namespaceing with camera topic.')
        
        linear_velocity_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Linear velocity for vehicle motion (m/s).')

        angular_velocity_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Angular velocity for vehicle motion (rad/s).')

        single_line_steer_scale_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Single found line steer scaling.')
        
        self.declare_parameter("start_delay",15.00, 
            start_delay_descriptor)
        
        self.declare_parameter("camera_vector_topic", "/cupcar0/PixyVector", 
            camera_vector_topic_descriptor)
        
        self.declare_parameter("linear_velocity", 1.25, 
            linear_velocity_descriptor)

        self.declare_parameter("angular_velocity", 1.5, 
            angular_velocity_descriptor)

        self.declare_parameter("single_line_steer_scale", 0.5, 
            single_line_steer_scale_descriptor)

        self.start_delay = float(self.get_parameter("start_delay").value)

        self.camera_vector_topic = str(self.get_parameter("camera_vector_topic").value)

        self.linear_velocity = float(self.get_parameter("linear_velocity").value)

        self.angular_velocity = float(self.get_parameter("angular_velocity").value)

        self.single_line_steer_scale = float(self.get_parameter("single_line_steer_scale").value)

        self.camera_vector_topic = '/cupcar0/PixyVector'
        self.cameraImageTopic = '/trackImage0/image_raw'

        # Time to wait before running
        self.get_logger().info('Waiting to start for {:s}'.format(str(self.start_delay)))
        sleep(self.start_delay)
        self.get_logger().info('Started')


        self.bridge = CvBridge()
 
    
        self.start_time = datetime.now().timestamp()
        self.restart_time = True
        self.passedImage = cv2.imread("/home/greyhat/ros2ws/build/aim_line_follow/aim_line_follow/passed.jpeg")
        
        # Subscribers
        self.pose_subscriber = self.create_subscription(
            Odometry,
            '/cupcar0/odom',
            self.odom_callback,
            10)



        self.pixy_subscriber = self.create_subscription(
            PixyVector,
            self.camera_vector_topic,
            self.listener_callback,
            0)





        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cupcar0/cmd_vel', 10)

        self.speed_vector = Vector3()
        self.steer_vector = Vector3()
        self.cmd_vel = Twist()
        self.position = Point()
        self.reset_request = Empty.Request()
        self.pause_request = Empty.Request()
        self.unpause_request = Empty.Request()
       
        
        self.new_episode = True
        self.prev_state = None


        self.prev_pose = Point()
        self.prev_pose.x = 0.0
        self.prev_pose.y = 0.0
        self.prev_pose.z = 0.394
        

        self.speed = 0
        self.steer = 0



        self.rewards = 0

        self.iteration = 0
        self.currT = time.time()






        
    def odom_callback(self,msg):
        self.position = msg.pose.pose.position


    def get_num_vectors(self, msg):
        scene = self.bridge.imgmsg_to_cv2(msg.raw_image, "bgr8")
        self.passedImage = copy.deepcopy(scene)
        num_vectors = 0
        if(not(msg.m0_x0 == 0 and msg.m0_x1 == 0 and msg.m0_y0 == 0 and msg.m0_y1 == 0)):
            num_vectors = num_vectors + 1
        if(not(msg.m1_x0 == 0 and msg.m1_x1 == 0 and msg.m1_y0 == 0 and msg.m1_y1 == 0)):
            num_vectors = num_vectors + 1
        return num_vectors


    def distance_moved(self,present_pose,prev_pose):
        diff_x = present_pose.x - prev_pose.x
        diff_y = present_pose.y - prev_pose.y
        distanceMoved = np.sqrt(np.square(diff_x)+np.square(diff_y) )
        return distanceMoved


         
    def resetWorld(self):
        reset_gazebo_world= self.create_client(Empty, '/reset_world')
        while not reset_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        reset_gazebo_world.call_async(self.reset_request)


    def pauseWorld(self):
        pause_gazebo_world= self.create_client(Empty, '/pause_physics')
        while not pause_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        pause_gazebo_world.call_async(self.pause_request)


    def playWorld(self):
        unpause_gazebo_world= self.create_client(Empty, '/unpause_physics')
        while not unpause_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        unpause_gazebo_world.call_async(self.unpause_request)



    def listener_callback(self, msg):

     
        if (self.iteration % 200 == 0 and self.iteration != 0):
            agent.save_model(self.iteration)
            for i in range(10):
                print(f" 200 iter in {(time.time() - self.currT)/60} min ")
                self.currT = time.time()
        num_vectors = self.get_num_vectors(msg)
        breakFlag = False

        self.pauseWorld()

        if(num_vectors == 0):
            # cv2.imwrite(f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",self.passedImage)
            # f = self.drive.CreateFile({'title': f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
            # f.SetContentFile(f"{os.environ['HOME']}/ros2ws/src/aim_line_follow/aim_line_follow/{self.new_episode}_{self.iteration}_{num_vectors}.jpeg")
            # f.Upload()
#             <tf.Tensor: shape=(1, 1), dtype=float32,
#  numpy=array([[10.228476]], dtype=float32)>, <tf.Tensor:
#  shape=(1, 1), dtype=float32, numpy=array([[0.44765955]],
#  dtype=float32)>]
# (<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array(
# [[2891.2368]], dtype=float32)>, <tf.Tensor: shape=(1, 1), 
# dtype=float32,
#  numpy=array([[2.966984e+14]], dtype=float32)>, <tf.Ten
# sor: shape=(1
# , 1), dtype=float32, numpy=array([[1.2477877e+10]], dtype=f
# loat32)>)

            self.iteration += 1
            self.rewards = 0
            self.resetWorld()
            self.speed = 0.0
            self.steer = 0.0
            self.new_episode = True
            self.prev_pose.x = 0.0
            self.prev_pose.y = 0.0
            self.prev_pose.z = 0.394

            self.speed_vector.x = float(self.speed)
            self.steer_vector.z = float(self.steer)

            self.cmd_vel.linear = self.speed_vector
            self.cmd_vel.angular = self.steer_vector

            self.cmd_vel_publisher.publish(self.cmd_vel)

            
            self.playWorld()
            tpassedImage = np.array(self.passedImage)
            tpassedImage = cv2.resize(tpassedImage,(64,64))
            tpassedImage = np.reshape(tpassedImage,(1,64,64,3))
            
            # reward = -1500
            # value_loss , speed_loss , steer_loss = agent.learn(reward , tpassedImage ,1 )
            # print(f"value_loss: {value_loss}, speed_loss:{speed_loss} , steer_loss:{steer_loss}")
            # print(self.iteration)

        if(num_vectors == 1):
            
            # cv2.imwrite(f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",self.passedImage)
            # f = self.drive.CreateFile({'title': f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
            # f.SetContentFile(f"{os.environ['HOME']}/ros2ws/src/aim_line_follow/aim_line_follow/{self.new_episode}_{self.iteration}_{num_vectors}.jpeg")
            # f.Upload()


            self.iteration += 1

            self.resetWorld()
            self.speed = 0.0
            self.steer = 0.0

            self.rewards = 0
            self.new_episode = True
            self.prev_pose.x = 0.0
            self.prev_pose.y = 0.0
            self.prev_pose.z = 0.394

            self.speed_vector.x = float(self.speed)
            self.steer_vector.z = float(self.steer)

            self.cmd_vel.linear = self.speed_vector
            self.cmd_vel.angular = self.steer_vector

            self.cmd_vel_publisher.publish(self.cmd_vel)

            # while(not(self.position.x * self.position.y == 0)):
            #         pass

            self.playWorld()
            tpassedImage = np.array(self.passedImage)
            tpassedImage = cv2.resize(tpassedImage,(64,64))
            tpassedImage = np.reshape(tpassedImage,(1,64,64,3))
            
            # reward = -1500
            # value_loss , speed_loss , steer_loss = agent.learn(reward , tpassedImage ,1 )
            # print(f"value_loss: {value_loss}, speed_loss:{speed_loss} , steer_loss:{steer_loss}")
            # print(self.iteration)
                        	
            	

        if(num_vectors == 2):
            if(self.new_episode): 
                # cv2.imwrite(f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",self.passedImage)
                # f = self.drive.CreateFile({'title': f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
                # f.SetContentFile(f"{os.environ['HOME']}/ros2ws/src/aim_line_follow/aim_line_follow/{self.new_episode}_{self.iteration}_{num_vectors}.jpeg")
                # f.Upload()
                img = np.array(self.passedImage)
                img = cv2.resize(img,(64,64))
                img = np.reshape(img,(1,64,64,3))
                # self.prev_state = self.passedImage

                tspeed,tsteer = agent.act(img)
                
               
                # while(True):
                #     file_list = self.drive.ListFile({'q': "'117e8JrRen96BCfhim0UynjpJOtbt88Uc' in parents and trashed=false", }).GetList()
                #     for file1 in file_list:
                #         if(file1['title'] == f"{self.new_episode}_{self.iteration}_{num_vectors}.txt"):
                #             file_dnld = self.drive.CreateFile({'id': file1['id']})
                #             file_dnld.GetContentFile(f"{self.new_episode}_{self.iteration}_{num_vectors}.txt", mimetype='text/plain')
                #             file_r = open(f"{self.new_episode}_{self.iteration}_{num_vectors}.txt","r")
                #             a = file_r.readline()
                #             self.speed,self.steer = [float(x) for x in a.split(" ")]
                #             breakFlag = True
                #             break
                #     if(breakFlag):
                #         breakFlag = False
                #         break
                self.speed,self.steer = float(tspeed),float(tsteer)

                # breakFlag = True

                self.new_episode = False
                self.iteration += 1

                self.playWorld()
                sleep(0.1) 

                self.speed_vector.x = float(self.speed)
                self.steer_vector.z = float(self.steer)

                self.cmd_vel.linear = self.speed_vector
                self.cmd_vel.angular = self.steer_vector

                self.cmd_vel_publisher.publish(self.cmd_vel)
                print(self.iteration)

                
                



            else:
                reward = -1 + self.distance_moved(self.position,self.prev_pose)
                # file1 = self.drive.CreateFile({'title': f"{self.new_episode}_{self.iteration}_{num_vectors}R.txt" , 'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
                # file1.SetContentString(f"{reward}")
                # file1.Upload()
        
                # cv2.imwrite(f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",self.passedImage)
                # f = self.drive.CreateFile({'title': f"{self.new_episode}_{self.iteration}_{num_vectors}.jpeg",'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
                # f.SetContentFile(f"{os.environ['HOME']}/ros2ws/src/aim_line_follow/aim_line_follow/{self.new_episode}_{self.iteration}_{num_vectors}.jpeg")
                # f.Upload()
                img = np.array(self.passedImage)
                img = cv2.resize(img,(64,64))
                img = np.reshape(img,(1,64,64,3))
                # value_loss , speed_loss , steer_loss = agent.learn(reward , img ,0 )
                # print(f"value_loss: {value_loss}, speed_loss:{speed_loss} , steer_loss:{steer_loss}")

                tspeed,tsteer = agent.act(img)
                self.speed,self.steer = float(tspeed),float(tsteer)

                # self.prev_state = self.passedImage
                self.new_episode = False
                self.prev_pose = self.position
                self.iteration += 1
                print(self.iteration)

                self.playWorld() 
                sleep(0.1)
                

                self.speed_vector.x = float(self.speed)
                self.steer_vector.z = float(self.steer)

                self.cmd_vel.linear = self.speed_vector
                self.cmd_vel.angular = self.steer_vector

                self.cmd_vel_publisher.publish(self.cmd_vel)

                

                


            
 



def main(args=None):
    rclpy.init(args=args)

    line_follow = LineFollow()

    rclpy.spin(line_follow)

    line_follow.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
