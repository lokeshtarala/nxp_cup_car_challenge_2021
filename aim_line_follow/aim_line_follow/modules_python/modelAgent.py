import numpy as np
from tensorflow.keras.models import Sequential
#import keras
import tensorflow as tf
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


def normalSampling(var):
    normal_dist =  tfp.distributions.Normal(loc=var[0], scale=var[1])
    sampled = tf.squeeze(normal_dist.sample(1), axis=0)
    return sampled

def clipSteer(val):
    return  tf.clip_by_value(val,-5,5)
def clipSpeed(val):
    return  tf.clip_by_value(val,0,100)

def value_function():
    model = Sequential()
    model.add(InputLayer(input_shape=(64,64,3)))
    #1*1 layer
    model.add(Conv2D(3,1,activation="elu"))
    
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(32,3,activation="elu"))
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(32,3,activation="elu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    
    
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64,3,activation="elu"))
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64,3,activation="elu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128,3,activation="elu"))
    #model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128,3,activation="elu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    #model.add(Dense(1024,activation = "elu"))
    model.add(Dense(512,activation = "elu"))
    model.add(Dense(64,activation = "elu"))
    model.add(Dense(16,activation = "elu"))
    
    #output steer and speed
    model.add(Dense(1,activation = "elu"))
    
    
    
    return model


def policy():
    X_input=Input(shape = (64,64,3))
    
    x = Conv2D(3,1,activation="elu")(X_input)
    
    x = Conv2D(32,3,activation="elu")(x)
    x = Conv2D(32,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(64,3,activation="elu")(x)
    x = Conv2D(64,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(128,3,activation="elu")(x)
    x = Conv2D(128,3,activation="elu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    xSplit = Dense(512,activation = "elu")(x)
    #xSplit = Dense(64,activation = "elu")(x)
    
    #steer
    xSteer = Dense(64,activation = "elu")(xSplit)
    xSteer = Dense(16,activation = "elu")(xSteer)
    muSteer = Dense(1,activation=None,name = "muSteer")(xSteer)
    sigmaSteer =  Dense(1,activation="softplus")(xSteer)
    sigmaSteer = Add(name = "sigmaSteer")([sigmaSteer,tf.constant([1e-5])])
    #steerDist =  tfp.distributions.Normal(muSteer, sigmaSteer,name = "speedDist")
    steer = Lambda(normalSampling,1)((muSteer,sigmaSteer))
    steer = Lambda(clipSteer,1,name = "steer")(steer)
    #speed
    xSpeed = Add()([steer,xSplit])
    xSpeed = Dense(64,activation = "elu")(xSplit)
    xSpeed = Dense(16,activation = "elu")(xSpeed)
    muSpeed = Dense(1,activation=None,name = "muSpeed")(xSpeed)
    sigmaSpeed =  Dense(1,activation="softplus")(xSpeed)
    sigmaSpeed = Add(name = "sigmaSpeed")([sigmaSpeed,tf.constant([1e-5])])
    #speedDist =  tfp.distributions.Normal(muSpeed, sigmaSpeed,name = "steerDist")
    speed = Lambda(normalSampling,1)((muSpeed,sigmaSpeed))
    speed = Lambda(clipSpeed,1,name="speed")(speed)
    
    model=Model(inputs=X_input,outputs =[speed,steer,muSpeed,sigmaSpeed,muSteer,sigmaSteer])
    return model 
    

