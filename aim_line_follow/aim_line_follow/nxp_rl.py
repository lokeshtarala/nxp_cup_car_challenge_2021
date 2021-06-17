import os
import cv2
import numpy as np


import numpy as np
from tensorflow.keras.models import Sequential
import keras
import tensorflow as tf

#for memory growth 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


from tensorflow.keras.layers import Layer,InputLayer,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,MaxPool2D,Dropout,Reshape,Add,Conv2DTranspose,Concatenate,Lambda
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
from tensorflow.python.keras.engine.input_layer import Input
import tensorflow_probability as tfp
import time


@tf.function
def normalSampling(var):
    normal_dist =  tfp.distributions.Normal(loc=var[0], scale=var[1])
    sampled = tf.squeeze(normal_dist.sample(1,seed = 123), axis=0)
    return sampled

def clipSteer(val):
    return  tf.clip_by_value(val,-0.5,0.5)
def clipSpeed(val):
    return  tf.clip_by_value(val,0.3,1)

def value_function():
    model = Sequential()
    model.add(InputLayer(input_shape=(64,64,3)))
    #1*1 layer
    model.add(Conv2D(3,1,activation="elu"))
    
    model.add(ZeroPadding2D(padding=(1,1)))

    model.add(Conv2D(32,3,activation="elu"))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(32,3,activation="elu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    
    
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64,3,activation="elu"))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64,3,activation="elu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128,3,activation="elu"))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128,3,activation="elu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(16,1,activation="elu"))    
    
    model.add(Flatten())
    
    #model.add(Dense(1024,activation = "elu"))
    model.add(Dense(128,activation = "elu"))
    model.add(Dense(64,activation = "elu"))
    model.add(Dense(16,activation = "elu"))
    
    #output steer and speed
    model.add(Dense(1,activation = "elu"))
    
    
    
    return model

def policy_steer():
    X_input=Input(shape = (64,64,3))
    
    x = Conv2D(3,1,activation="elu")(X_input)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(32,3,activation="elu")(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(32,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(64,3,activation="elu")(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(64,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(128,3,activation="elu")(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(128,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    

    x = Conv2D(16,1,activation="elu")(x)

    x = Flatten()(x)
    x = Dense(128,activation = "elu")(x)
    #steer
    x = Dense(64,activation = "elu")(x)
    x = Dense(16,activation = "elu")(x)
    muSteer = Dense(1,activation="tanh",name = "muSteer")(x)
    sigmaSteer =  Dense(1,activation="softplus")(x)
    sigmaSteer = Add(name = "sigmaSteer")([sigmaSteer,tf.constant([1e-5])])
 
    model=Model(inputs=X_input,outputs =[muSteer,sigmaSteer])
    return model


def policy_speed():
    X_input=Input(shape = (64,64,3))
    steer_input = Input(shape = (1))
    
    x = Conv2D(3,1,activation="elu")(X_input)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(32,3,activation="elu")(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(32,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(64,3,activation="elu")(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(64,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(128,3,activation="elu")(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(128,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    

    x = Conv2D(16,1,activation="elu")(x)

    x = Flatten()(x)
    x = Dense(128,activation = "elu")(x)
    x = Add()([x, steer_input])
    #steer
    x = Dense(64,activation = "elu")(x)
    x = Dense(16,activation = "elu")(x)
    muSpeed = Dense(1,activation="relu",name = "muSteer")(x)
    sigmaSpeed =  Dense(1,activation="softplus")(x)
    sigmaSpeed = Add(name = "sigmaSteer")([sigmaSpeed,tf.constant([1e-5])])
 
    model=Model(inputs=[X_input,steer_input],outputs =[muSpeed,sigmaSpeed])
    return model

def policy_loss(y_pred , action , td):
  normal_dist =  tfp.distributions.Normal(loc=y_pred[0], scale=y_pred[1])
  return -1 * normal_dist.log_prob(action) * td

class Agent():
  def __init__(self,value_model,speed_model,steer_model,load_model = False):
    self.value_model = value_model
    self.speed_model = speed_model
    self.steer_model = steer_model
    self.actions = None
    self.prev_state = None
    if(load_model):
      self.value_model.load_weights("/home/greyhat/ros2ws/build/aim_line_follow/aim_line_follow/value_model_1400.hdf5")
      self.speed_model.load_weights("/home/greyhat/ros2ws/build/aim_line_follow/aim_line_follow/speed_model_1400.hdf5")
      self.steer_model.load_weights("/home/greyhat/ros2ws/build/aim_line_follow/aim_line_follow/steer_model_1400.hdf5")
      


  def save_model(self,iteration):
    self.value_model.save_weights(f"value_model_{iteration}.hdf5")
    self.speed_model.save_weights(f"speed_model_{iteration}.hdf5")
    self.steer_model.save_weights(f"steer_model_{iteration}.hdf5")


  def policy_loss(self,y_pred , action , td):
    normal_dist =  tfp.distributions.Normal(loc=y_pred[0], scale=y_pred[1])
    return -1 * normal_dist.log_prob(action) * td

  def act(self, state):
    steer_probs = self.steer_model.predict(state)
    steer_dist = tfp.distributions.Normal(loc=steer_probs[0], scale=steer_probs[1])
    steer = tf.squeeze(steer_dist.sample(1,seed = 123), axis=0)

    speed_probs = self.speed_model.predict([state,steer])
    speed_dist = tfp.distributions.Normal(loc=speed_probs[0], scale=speed_probs[1])
    speed = tf.squeeze(speed_dist.sample(1,seed = 123), axis=0)

    self.actions = [speed,steer]
    self.prev_state = state

    return [speed,steer]



  def learn(self, reward , state , done, gamma = 0.99):
    with tf.GradientTape() as tape1 , tf.GradientTape() as tape2 ,tf.GradientTape() as tape3 :
      
      prev_V = self.value_model(self.prev_state,training = True)
      current_V = self.value_model(state , training = False)
      td = reward + gamma*current_V*(1- int (done)) - prev_V
    
      speed_ps = self.speed_model(inputs =[self.prev_state,self.actions[1]] ,training = True)
      steer_ps = self.steer_model(self.prev_state, training = True)

      value_loss = td**2
      speed_loss = self.policy_loss(speed_ps, self.actions[0] , td)
      steer_loss = self.policy_loss(steer_ps, self.actions[1], td)


    value_grads = tape1.gradient(value_loss ,self.value_model.trainable_variables)
    speed_grads = tape2.gradient(speed_loss ,self.speed_model.trainable_variables)
    steer_grads = tape3.gradient(steer_loss ,self.steer_model.trainable_variables)

    opt1 = tf.keras.optimizers.Adam(learning_rate=0.0001)
    opt2 = tf.keras.optimizers.Adam(learning_rate=0.0001)
    opt3 = tf.keras.optimizers.Adam(learning_rate=0.0001)

    opt1.apply_gradients(zip(value_grads,self.value_model.trainable_variables))
    opt2.apply_gradients(zip(speed_grads,self.speed_model.trainable_variables))
    opt3.apply_gradients(zip(steer_grads,self.steer_model.trainable_variables))

    return value_loss , speed_loss , steer_loss

# iteration = 0
# prev_state = None
# gamma = 0.99
# speed,steer  = None ,None
agent = Agent(value_function(),policy_speed(),policy_steer(),load_model = True)

# img = cv2.imread("/home/greyhat/passed.jpeg")
# tdf = time.time()
# img = np.array(img)
# img = cv2.resize(img,(64,64))
# img = np.reshape(img,(1,64,64,3))
# speed,steer = agent.act(img)
# print(float(speed))
# # agent.learn(54,img,0)
# print(time.time() - tdf)

# while(True):

#   if (iteration % 200 == 0 and iteration != 0):

#     agent.save_model(iteration)

#   if (os.path.isfile(f"/content/drive/MyDrive/RL4NXP/True_{iteration}_2.jpeg")):

#     img = cv2.imread(f"/content/drive/MyDrive/RL4NXP/True_{iteration}_2.jpeg")
#     img = np.array(img)
#     img = cv2.resize(img,(64,64))
#     img = np.reshape(img,(1,64,64,3))


#     speed,steer = agent.act(img)
    
#     file1 = drive.CreateFile({'title': f"True_{iteration}_2.txt" , 'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
#     file1.SetContentString(f"{float(speed)} {float(steer)}")
#     file1.Upload()

#     # prev_state = img 
#     iteration += 1
#     print(iteration)

#   elif (os.path.isfile(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_2.jpeg")):

#     img = cv2.imread(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_2.jpeg")
#     img = np.array(img)
#     img = cv2.resize(img,(64,64))
#     img = np.reshape(img,(1,64,64,3))

#     # prev_V = value_model.predict(prev_state)
#     # current_V = value_model.predict(img)
    
#     file1 = open(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_2R.txt","r")
#     reward = float (file1.readline())

#     # target = reward + gamma*current_V
#     # td_error = target - prev_V
#     # value_model.fit(x= prev_state, y = target)

#     value_loss , speed_loss , steer_loss = agent.learn(reward , img ,0 )
#     print(f"value_loss: {value_loss}, speed_loss:{speed_loss} , steer_loss:{steer_loss}")

#     speed,steer = agent.act(img)
    
#     file1 = drive.CreateFile({'title': f"False_{iteration}_2.txt" , 'parents':[{'id':'117e8JrRen96BCfhim0UynjpJOtbt88Uc'}]})
#     file1.SetContentString(f"{float(speed)} {float(steer)}")
#     file1.Upload()

#     iteration += 1
#     print(iteration)


#   elif (os.path.isfile(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_0.jpeg")):

#     passedImage = cv2.imread(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_0.jpeg")
#     passedImage = np.array(passedImage)
#     passedImage = cv2.resize(passedImage,(64,64))
#     passedImage = np.reshape(passedImage,(1,64,64,3))

#     # prev_V = value_model.predict(prev_state)
#     # current_V = value_model.predict(passedImage)

#     reward = -1500
#     # target = reward + gamma*current_V
#     # td_error = target - prev_V
#     # value_model.fit(x= prev_state, y = target)

#     value_loss , speed_loss , steer_loss = agent.learn(reward , passedImage ,1 )
#     print(f"value_loss: {value_loss}, speed_loss:{speed_loss} , steer_loss:{steer_loss}")
#     speed,steer = None,None

#     iteration += 1
#     print(iteration)


#   elif (os.path.isfile(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_1.jpeg")):

#     passedImage = cv2.imread(f"/content/drive/MyDrive/RL4NXP/False_{iteration}_1.jpeg")
#     passedImage = np.array(passedImage)
#     passedImage = cv2.resize(passedImage,(64,64))
#     passedImage = np.reshape(passedImage,(1,64,64,3))

#     reward = -1500
#     value_loss , speed_loss , steer_loss = agent.learn(reward , passedImage ,1 )
#     print(f"value_loss: {value_loss}, speed_loss:{speed_loss} , steer_loss:{steer_loss}")
#     speed,steer = None,None


#     speed,steer = None,None
#     iteration += 1
#     print(iteration)
