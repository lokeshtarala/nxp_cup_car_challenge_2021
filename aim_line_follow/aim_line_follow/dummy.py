#!/usr/bin/env python3

from re import L
import rclpy
from rclpy.node import Node

from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from nxp_cup_interfaces.msg import PixyVector
from time import sleep
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from gazebo_msgs.srv import GetModelState

import sys 
import os
from .modules_python import modelAgent

from .modules_python import simulationSettings


import math

import numpy as np
from tensorflow.keras.models import Sequential
#import keras

#only cpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

from tensorflow.keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,MaxPool2D,Dropout,Reshape,Add,Conv2DTranspose,Concatenate,Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import tensorflow_probability as tfp
import copy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Point
from rclpy.task import Future
from std_srvs.srv import Empty
from cv_bridge.core import CvBridge


#nav_msgs/msg/Odometry




class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name

class Tutorial:

    _blockListDict = {
        'block_a': Block('nxp_cupcar', 'front_left_wheel_link')

    }

    def show_gazebo_models(self):
         model_coordinates = Node.create_client('/gazebo/get_model_state',GetModelState)
         #resp = add_two_ints.call_async(req)
         #model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
         for block in self._blockListDict.itervalues():
             blockName = str(block._name)
             resp_coordinates = model_coordinates(blockName, block._relative_entity_name)
             print(blockName)
             print("Cube " + str(block._name))
             print("Valeur de X : " + str(resp_coordinates.pose.position.x))
             print("Quaternion X : " + str(resp_coordinates.pose.orientation.x))
             



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

        self.start_time = datetime.now().timestamp()
        self.restart_time = True
        self.passedImage = cv2.imread("/home/greyhat/Pictures/test1.png")

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
            10)

        self.imageSub = self.create_subscription( Image, 
                    '/trackImage0/image_raw', 
                    self.pixyImageCallback, 
                    qos_profile_sensor_data)
                #/trackImage0/image_raw



        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cupcar0/cmd_vel', 10)

        self.speed_vector = Vector3()
        self.steer_vector = Vector3()
        self.cmd_vel = Twist()
        self.position = Point()
        self.reset_request = Empty.Request()
        self.pause_request = Empty.Request()
        self.unpause_request = Empty.Request()
        self.bridge = CvBridge()
    
        
        self.new_episode = True
        self.prev_state = None
        self.prev_pose = Point()
        self.prev_pose.x = 0.0
        self.prev_pose.y = 0.0
        self.prev_pose.z = 0.394
        
    
        #self.value_model = modelAgent.value_function()
        #self.value_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.MeanSquaredError())
        

        self.speed = 0
        self.steer = 0
        self.muSpeed = 0
        self.muSteer = 0
        self.sigmaSpeed = 0
        self.sigmaSteer = 0
        self.td_error = 0

        self.rewards = 0
        self.policy_model = modelAgent.policy()
        
    def odom_callback(self,msg):

        self.position = msg.pose.pose.position
        # print(f"present position: {self.position}")


    

    def pixyImageCallback(self,msg):
         # Scene from subscription callback
        scene = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        #deep copy and pyramid down image to reduce resolution
        scenePyr = copy.deepcopy(scene)
        if self.pyrDown > 0:
            for i in range(self.pyrDown):
                scenePyr = cv2.pyrDown(scenePyr)
        self.passedImage = copy.deepcopy(scenePyr)
        print(np.shape(self.passedImage))

    def get_num_vectors(self, msg):
        num_vectors = 0
        if(not(msg.m0_x0 == 0 and msg.m0_x1 == 0 and msg.m0_y0 == 0 and msg.m0_y1 == 0)):
            num_vectors = num_vectors + 1
        if(not(msg.m1_x0 == 0 and msg.m1_x1 == 0 and msg.m1_y0 == 0 and msg.m1_y1 == 0)):
            num_vectors = num_vectors + 1
        return num_vectors

    # def timer_callback(self):
    #     #TODO


    def distance_moved(self,present_pose,prev_pose):
        disp = list() 
        diff_x = present_pose.x - prev_pose.x
        diff_y = present_pose.y - prev_pose.y
        diff_z = present_pose.z - prev_pose.z
        # disp.append(diff_x)
        # disp.append(diff_y)
        # disp.append(diff_z)

        distanceMoved = np.sqrt(np.square(diff_x)+np.square(diff_y)+np.square(diff_z))   # use numpy
        return distanceMoved


         

    def policy_loss(self,mu,sigma,td):
        def keras_loss(y_true,y_pred):
            normal_dist = tfp.distributions.Normal(mu,sigma)
            return -1* tf.math.log(normal_dist.prob(y_pred)+1e-5)*td
        return keras_loss

    def resetworld(self):
        reset_gazebo_world= self.create_client(Empty, '/reset_world')
        while not reset_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        resp = reset_gazebo_world.call_async(self.reset_request)
        rclpy.spin_until_future_complete(self,resp)

    def pauseWorld(self):
        pause_gazebo_world= self.create_client(Empty, '/pause_physics')
        while not pause_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        resp = pause_gazebo_world.call_async(self.pause_request)
        rclpy.spin_until_future_complete(self,resp)

    def playWorld(self):
        unpause_gazebo_world= self.create_client(Empty, '/unpause_physics')
        while not unpause_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        resp = unpause_gazebo_world.call_async(self.unpause_request)
        rclpy.spin_until_future_complete(self,resp)


    def listener_callback(self, msg):
        #TODO
        current_time = datetime.now().timestamp()
        frame_width = 79
        frame_height = 52
        window_center = (frame_width / 2)
        x = 0
        y = 0
        steer = 0
        speed = 0
        num_vectors = self.get_num_vectors(msg)
        self.passedImage = np.array(self.passedImage)
        print("image converted to numpy array")
        self.passedImage = cv2.resize(self.passedImage,(64,64))
        self.passedImage = np.reshape(self.passedImage,(1,64,64,3))
        print("shape of image= 0",np.shape(self.passedImage))

        gamma = 0.99
        print("going to pause the world")
        self.pauseWorld()
        print("paused the world")
        if(num_vectors == 0):
            print("in number of vectors ==0")
            reward = -1500
            value_model = modelAgent.value_function()
            value_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.MeanSquaredError())
        
            prev_V = value_model.predict(self.prev_state)
            current_V = value_model.predict(self.passedImage)
            target = reward + gamma*current_V
            self.td_error = target - prev_V
            value_model.fit(x= self.prev_state, y = target)
            losses={'speed':self.policy_loss(self.muSpeed,self.sigmaSpeed,self.td_error),'steer':self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"muSpeed" :self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"sigmaSpeed" :self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"muSteer":policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"sigmaSteer":policy_loss(self.muSteer,self.sigmaSteer,self.td_error)}
            lossWeights={'speed':1,'steer':1,"muSpeed":0,"sigmaSpeed":0,"muSteer":0,"sigmaSteer":0}
            self.policy_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=losses,loss_weights=lossWeights)
            self.policy_model.fit(self.prev_state,[np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1)])
            #reset gazebo
            self.rewards = 0
            self.resetWorld()
            self.new_episode = True
            self.prev_pose.x = 0
            self.prev_pose.y = 0
            self.prev_pose.z = 0.394

        if(num_vectors == 1):
            print("number of vectors==1 ")
            reward = -1500
            value_model = modelAgent.value_function()
            value_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.MeanSquaredError())
        
            prev_V = value_model.predict(self.prev_state)
            current_V = value_model.predict(self.passedImage)
            target = reward + gamma*current_V
            self.td_error = target - prev_V
            value_model.fit(x= self.prev_state, y = target)
            losses={'speed':self.policy_loss(self.muSpeed,self.sigmaSpeed,self.td_error),'steer':self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"muSpeed" :self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"sigmaSpeed" :self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"muSteer":self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"sigmaSteer":self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error)}
            lossWeights={'speed':1,'steer':1,"muSpeed":0,"sigmaSpeed":0,"muSteer":0,"sigmaSteer":0}
            self.policy_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss=losses,loss_weights=lossWeights)
            self.policy_model.fit(self.prev_state,[np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1)])
            #reset gazebo
            
            self.resetworld()
            self.rewards = 0
            self.new_episode = True
            self.prev_pose.x = 0
            self.prev_pose.y = 0
            self.prev_pose.z = 0.394
                        	
            	

        if(num_vectors == 2):
            print("in number of vectors==2")
            if(self.new_episode):
                print("going to preidct")
                self.prev_state = self.passedImage
                self.speed,self.steer,self.muSpeed,self.sigmaSpeed,self.muSteer,self.sigmaSteer = self.policy_model.predict(self.passedImage)
                print("predicted")
                self.new_episode = False

            else:
                reward = -1 + self.distance_moved(self.position,self.prev_pose)
                value_model = modelAgent.value_function()
                value_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.MeanSquaredError())
        
                prev_V = value_model.predict(self.prev_state)
                current_V = value_model.predict(self.passedImage)


                target = reward + gamma*current_V
                self.td_error = target - prev_V
                value_model.fit(x= self.prev_state, y = target)
                losses={'speed':self.policy_loss(self.muSpeed,self.sigmaSpeed,self.td_error),'steer':self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"muSpeed" :self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"sigmaSpeed" :self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"muSteer":self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error),"sigmaSteer":self.policy_loss(self.muSteer,self.sigmaSteer,self.td_error)}
                lossWeights={'speed':1,'steer':1,"muSpeed":0,"sigmaSpeed":0,"muSteer":0,"sigmaSteer":0}
                self.policy_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss=losses,loss_weights=lossWeights)
                self.policy_model.fit(self.prev_state,[np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1),np.array(0.0).reshape(-1,1)])
                self.speed,self.steer,self.muSpeed,self.sigmaSpeed,self.muSteer,self.sigmaSteer = self.policy_model.predict(self.passedImage)
                self.prev_state = self.passedImage
                self.new_episode = False
                self.prev_pose = self.position

        
        self.playWorld() 
            
        self.rewards += reward
        print(self.rewards)   
        self.speed_vector.x = float(self.speed)
        self.steer_vector.z = float(self.steer)

        self.cmd_vel.linear = self.speed_vector
        self.cmd_vel.angular = self.steer_vector

        self.cmd_vel_publisher.publish(self.cmd_vel)
        tuto = Tutorial()
        tuto.show_gazebo_models()

def main(args=None):
    rclpy.init(args=args)

    line_follow = LineFollow()

    rclpy.spin(line_follow)

    line_follow.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
