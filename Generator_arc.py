# Import all layer types from tensorflow library
from tensorflow.keras.layers import *
import tensorflow as tf

def Generator256():
    # weight initialization
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=[256,256,3])

    c = Conv2D(filters=64, kernel_size=7, padding='same', kernel_initializer=init)(input_layer) #(bs, 128, 128, 64)
    inst_norm = InstanceNormalization(axis=-1)(c)
    c = Activation('relu')(inst_norm)

    c = Conv2D(128,(3,3),strides=(2,2),padding='same', kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(c)
    ac = Activation('relu')(inst_norm)

    c = Conv2D(256, (3,3),strides=(2,2),padding='same',kernel_initializer=init)(ac)
    inst_norm = InstanceNormalization(axis=-1)(c)
    c = Activation('relu')(inst_norm)

    #Resnet
    for i in range(6):
        c = resnet_block(c,256)

    ct = Conv2DTranspose(128, (3,3), strides=(2,2),padding='same',kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(ct)
    c = Activation('relu')(inst_norm)

    ct = Conv2DTranspose(64, (3,3), strides=(2,2),padding='same',kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(ct)
    c = Activation('relu')(inst_norm)

    last = Conv2DTranspose(filters=3,kernel_size=(7,7), padding='same',kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(last)
    c = Activation('tanh')(inst_norm)

    return tf.keras.models.Model(inputs = input_layer,outputs=c)



def Generator128():
    # weight initialization
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=[128,128,3])

    c = Conv2D(filters=64, kernel_size=7, padding='same', kernel_initializer=init)(input_layer) #(bs, 128, 128, 64)
    inst_norm = InstanceNormalization(axis=-1)(c)
    c = Activation('relu')(inst_norm)

    c = Conv2D(128,(3,3),strides=(2,2),padding='same', kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(c)
    ac = Activation('relu')(inst_norm)

    c = Conv2D(256, (3,3),strides=(2,2),padding='same',kernel_initializer=init)(ac)
    inst_norm = InstanceNormalization(axis=-1)(c)
    c = Activation('relu')(inst_norm)

    #Resnet
    for i in range(6):
        c = resnet_block(c,256)

    ct = Conv2DTranspose(128, (3,3), strides=(2,2),padding='same',kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(ct)
    c = Activation('relu')(inst_norm)

    ct = Conv2DTranspose(64, (3,3), strides=(2,2),padding='same',kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(ct)
    c = Activation('relu')(inst_norm)

    last = Conv2DTranspose(filters=3,kernel_size=(7,7), padding='same',kernel_initializer=init)(c)
    inst_norm = InstanceNormalization(axis=-1)(last)
    c = Activation('tanh')(inst_norm)

    # return the model
    return tf.keras.models.Model(inputs = input_layer,outputs=c)
