# -*- coding: utf-8 -*-
"""
DL networks

Created on Tue Apr 21 2020
@author: Shihan Qiu (qiushihan19@gmail.com)
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D, Add, BatchNormalization, ReLU, LeakyReLU, ELU, PReLU, AveragePooling2D


class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


# ---------------- basic blocks ---------------- 
def ConvUnit(filters, kernel_size, strides, inputs, activation='relu', norm_type = 'batch'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # convolution
    conv = Conv2D(filters, kernel_size, strides, padding = 'same', kernel_initializer=initializer, use_bias=True)(inputs)
   
    # normalization
    if norm_type == 'batch':
        bn = BatchNormalization()(conv)
    elif norm_type == 'instance':
        bn = InstanceNormalization()(conv)

    # activation
    if activation == 'leakyrelu':
        outputs = LeakyReLU()(bn)
    elif activation == 'prelu':
        outputs = PReLU()(bn)
    elif activation == 'elu':
        outputs = ELU()(bn)
    elif activation == 'relu':
        outputs = ReLU()(bn)
    return outputs


def ConvGroup(filters, kernel_size, strides, nlayer, inputs, activation='relu', norm_type = 'batch'):
    x = inputs
    for i in range(nlayer):
        x = ConvUnit(filters, kernel_size, strides, x, activation)
    return x


def ConvTransUnit(filters, kernel_size, strides, inputs, activation='relu', norm_type = 'batch'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # deconvolution
    conv = Conv2DTranspose(filters, kernel_size, strides, padding = 'same', kernel_initializer=initializer, use_bias=True)(inputs)
   
    # normalization
    if norm_type == 'batch':
        bn = BatchNormalization()(conv)
    elif norm_type == 'instance':
        bn = InstanceNormalization()(conv)

    # activation
    if activation == 'leakyrelu':
        outputs = LeakyReLU()(bn)
    elif activation == 'prelu':
        outputs = PReLU()(bn)
    elif activation == 'elu':
        outputs = ELU()(bn)
    elif activation == 'relu':
        outputs = ReLU()(bn)
    return outputs

def UpConvUnit(filters, kernel_size, strides, inputs, activation='relu', norm_type = 'batch'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # upsampling
    up = UpSampling2D(size = strides)(inputs)
    # convolution
    conv = Conv2D(filters, kernel_size, strides=(1,1), padding = 'same', kernel_initializer=initializer, use_bias=True)(up)
   
    # normalization
    if norm_type == 'batch':
        bn = BatchNormalization()(conv)
    elif norm_type == 'instance':
        bn = InstanceNormalization()(conv)

    # activation
    if activation == 'leakyrelu':
        outputs = LeakyReLU()(bn)
    elif activation == 'prelu':
        outputs = PReLU()(bn)
    elif activation == 'elu':
        outputs = ELU()(bn)
    elif activation == 'relu':
        outputs = ReLU()(bn)
    return outputs

# ---------------- U-Net ---------------- 
def Unet_up(IN_CHANNEL,OUT_CHANNEL,CH1,NSTEP,NLAYER,pooling='max',activation='relu',norm_type='batch'):
    # NSTEP: number of encoder and decoder steps (number of resolution levels)
    # NLAYER: number of conv units in each resolution level

    h = inputs = Input(shape=[None, None, IN_CHANNEL])

    # encoder
    skips = []
    k = 1
    for i in range(NSTEP):
        # conv
        h = ConvGroup(filters=CH1*k, kernel_size=3, strides=1, nlayer=NLAYER, inputs=h, activation=activation, norm_type=norm_type)
        skips.append(h)
        # pooling
        if pooling == 'max':
            h = MaxPooling2D(pool_size=(2, 2))(h)
        elif pooling == 'average':
            h = AveragePooling2D(pool_size=(2, 2))(h)

        k = k*2

    # bottleneck
    h = ConvGroup(filters=CH1*k, kernel_size=3, strides=1, nlayer=NLAYER, inputs=h, activation=activation, norm_type=norm_type)

    # decoder
    for i in range(NSTEP):
        k = k/2
        # upconv and concatenation
        if pooling == 'max':
            h = UpSampling2D(size = (2,2), interpolation='nearest')(h)
        elif pooling == 'average':
            h = UpSampling2D(size = (2,2), interpolation='bilinear')(h)

        h = Conv2D(filters=CH1*k, kernel_size=2, padding = 'same', use_bias=True)(h)
        h = concatenate([skips[-i-1],h], axis = -1)
        # conv
        h = ConvGroup(filters=CH1*k, kernel_size=3, strides=1, nlayer=NLAYER, inputs=h, activation=activation, norm_type=norm_type)

    # output
    h = Conv2D(OUT_CHANNEL, 1)(h)
    model = Model(inputs = inputs, outputs = h)

    return model

# ---------------- ResNet ---------------- 
def Resblock(filters, kernel_size, strides, inputs, activation='relu', norm_type = 'batch', use_dropout=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    h = inputs

    # convolution
    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = Conv2D(filters, kernel_size, strides, padding = 'valid', kernel_initializer=initializer, use_bias=True)(h)
   
    # normalization
    if norm_type == 'batch':
        h = BatchNormalization()(h)
    elif norm_type == 'instance':
        h = InstanceNormalization()(h)

    # activation
    if activation == 'leakyrelu':
        h = LeakyReLU()(h)
    elif activation == 'prelu':
        h = PReLU()(h)
    elif activation == 'elu':
        h = ELU()(h)
    elif activation == 'relu':
        h = ReLU()(h)

    # drop out
    if use_dropout:
        h = Dropout(0.5)(h)

    # convolution
    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = Conv2D(filters, kernel_size, strides, padding = 'valid', kernel_initializer=initializer, use_bias=True)(h)

    # normalization
    if norm_type == 'batch':
        h = BatchNormalization()(h)
    elif norm_type == 'instance':
        h = InstanceNormalization()(h)

    outputs = Add()([inputs, h])

    return outputs



def Resnet(IN_CHANNEL,OUT_CHANNEL,CH1,Nres,activation='relu',norm_type='batch',use_dropout=False):
    h = inputs = Input(shape=[None, None, IN_CHANNEL])

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(CH1, 7, padding='valid', use_bias=True)(h)
    if norm_type == 'batch':
        h = BatchNormalization()(h)
    elif norm_type == 'instance':
        h = InstanceNormalization()(h)
    h = tf.nn.relu(h)

     # downsampling
    h = ConvUnit(CH1*2, 3, 2, h, activation, norm_type)
    h = ConvUnit(CH1*4, 3, 2, h, activation, norm_type)

    # residual blocks (Nres = number of residual blocks)
    for i in range(Nres):
        h = Resblock(CH1*4, 3, 1, h, activation, norm_type, use_dropout)

    # upsampling
    h = ConvTransUnit(CH1*2, 3, 2, h, activation, norm_type)
    h = ConvTransUnit(CH1, 3, 2, h, activation, norm_type)

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(OUT_CHANNEL, 7, 1, padding='valid')(h)
    #h = tf.tanh(h)

    model = Model(inputs = inputs, outputs = h)

    return model


