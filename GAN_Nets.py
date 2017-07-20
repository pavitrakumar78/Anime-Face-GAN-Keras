# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 12:41:47 2017

@author: Pavitrakumar
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import tqdm
from PIL import Image
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, merge
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from misc_layers import MinibatchDiscrimination, SubPixelUpscaling, CustomLRELU, bilinear2x
from keras_contrib.layers.convolutional import SubPixelUpscaling
from keras.datasets import mnist
import keras.backend as K
from keras.initializers import RandomNormal
K.set_image_dim_ordering('tf')


#import keras.backend as K
#K.set_learning_phase(1)

#we would have BatchNormalization layers on all but the generator output and discriminator input layers
 
np.random.seed(42)


def get_gen_normal(noise_shape):
    noise_shape = noise_shape
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    gen_input = Input(shape = noise_shape) #if want to directly use with conv layer next
    #gen_input = Input(shape = [noise_shape]) #if want to use with dense layer next
    
    generator = Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init)(gen_input)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
        
    #generator = bilinear2x(generator,256,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 256, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,128,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 128, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,64,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)    
    generator = Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,3,kernel_size=(3,3))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 3, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Activation('tanh')(generator)
        
    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(input = gen_input, output = generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model
    
#------------------------------------------------------------------------------------------

def get_disc_normal(image_shape=(64,64,3)):
    image_shape = image_shape
    
    dropout_prob = 0.4
    
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    dis_input = Input(shape = image_shape)
    
    discriminator = Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    discriminator = Flatten()(discriminator)
    
    #discriminator = MinibatchDiscrimination(100,5)(discriminator)
    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)
    
    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input = dis_input, output = discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    discriminator_model.summary()
    return discriminator_model

#------------------------------------------------------------------------------------------
