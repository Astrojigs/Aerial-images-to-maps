# Import all layer types from tensorflow library
from tensorflow.keras.layers import *
import tensorflow as tf


def Discriminator256():
    input_layer = Input(shape=[256,256,3])
    # downsampling
    conv1 = Conv2D(filters=64, kernel_size=4, strides=2,padding='same',activation='relu')(input_layer) # (bs, 64, 64, 64)

    leaky1 = LeakyReLU()(conv1)

    conv2 = Conv2D(filters=128,kernel_size=4, strides=2,activation='relu',padding='same')(leaky1) #(bs, 32, 32, 128)
    bat_norm = InstanceNormalization(axis=-1)(conv2)
    leaky2 = LeakyReLU()(bat_norm)

    conv3 = Conv2D(filters=256,kernel_size=4, strides=2,activation='relu',padding='same')(leaky2) #(bs, 16, 16, 256)
    bat_norm = InstanceNormalization(axis=-1)(conv3)
    leaky3 = LeakyReLU()(bat_norm)

    zero_pad1 = ZeroPadding2D()(leaky3)
    conv = Conv2D(filters=512, kernel_size=4, strides=1, use_bias=False)(zero_pad1)

    batch_norm = InstanceNormalization(axis=-1)(conv)

    leaky_relu = LeakyReLU()(batch_norm)

    zero_pad2 = ZeroPadding2D()(leaky_relu)

    last = Conv2D(1, 4, strides=1)(zero_pad2)

    return tf.keras.Model(inputs=input_layer, outputs=last)


def Discriminator128():
    input_layer = Input(shape=[128,128,3])
    # downsampling
    conv1 = Conv2D(filters=64, kernel_size=4, strides=2,padding='same',activation='relu')(input_layer) # (bs, 64, 64, 64)

    leaky1 = LeakyReLU()(conv1)

    conv2 = Conv2D(filters=128,kernel_size=4, strides=2,activation='relu',padding='same')(leaky1) #(bs, 32, 32, 128)
    bat_norm = InstanceNormalization(axis=-1)(conv2)
    leaky2 = LeakyReLU()(bat_norm)

    conv3 = Conv2D(filters=256,kernel_size=4, strides=2,activation='relu',padding='same')(leaky2) #(bs, 16, 16, 256)
    bat_norm = InstanceNormalization(axis=-1)(conv3)
    leaky3 = LeakyReLU()(bat_norm)

    zero_pad1 = ZeroPadding2D()(leaky3)
    conv = Conv2D(filters=512, kernel_size=4, strides=1, use_bias=False)(zero_pad1)

    batch_norm = InstanceNormalization(axis=-1)(conv)

    leaky_relu = LeakyReLU()(batch_norm)

    zero_pad2 = ZeroPadding2D()(leaky_relu)

    last = Conv2D(1, 4, strides=1)(zero_pad2)

    return tf.keras.Model(inputs=input_layer, outputs=last)
