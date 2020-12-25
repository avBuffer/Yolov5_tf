#! /usr/bin/env python
# coding=utf-8
import core.common as common
from core.mnv3_layers import *

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def darknet53(input_data, trainable):
    with tf.variable_scope('darknet'):
        # input_data = tf.reshape(input_data, [-1, 416, 416, 3]) # print layer's shape
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64), trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 64, 32, 64, trainable=trainable, name='residual%d' % (i + 0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128), trainable=trainable, name='conv4', downsample=True)
        for i in range(2):
            input_data = common.residual_block(input_data, 128, 64, 128, trainable=trainable, name='residual%d' % (i + 1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256), trainable=trainable, name='conv9', downsample=True)
        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' % (i + 3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' % (i + 11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' % (i + 19))

        return route_1, route_2, input_data


def mobilenetv2(input_data, trainable):
    with tf.variable_scope('mobilenetv2'):
        #input_data = tf.reshape(input_data, [-1, 416, 416, 3]) # print layer's shape

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0', downsample=True)
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 16), trainable=trainable, name='conv1', downsample=True)        
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 16, 24), trainable=trainable, name='conv2')
        
        for i in range(1):
            input_data = common.residual_block(input_data, 24, 24, 24, trainable=trainable, name='residual%d' % (i + 0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 24, 32), trainable=trainable, name='conv4', downsample=True)
        
        for i in range(2):
            input_data = common.residual_block(input_data, 32, 32, 32, trainable=trainable, name='residual%d' % (i + 1))
        
        route_1 = input_data


        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64), trainable=trainable, name='conv7', downsample=True)

        for i in range(3):
            input_data = common.residual_block(input_data, 64, 384, 64, trainable=trainable, name='residual%d' % (i + 3))


        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 96), trainable=trainable, name='conv11')

        for i in range(2):
            input_data = common.residual_block(input_data, 96, 576, 96, trainable=trainable, name='residual%d' % (i + 6))

        route_2 = input_data
        

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 96, 160), trainable=trainable, name='conv14', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 160, 160, 160, trainable=trainable, name='residual%d' % (i + 8))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 160, 320), trainable=trainable, name='conv17')   

        return route_1, route_2, input_data


def mobilenetv3(input_data, trainable):
    reduction_ratio = 4
    with tf.variable_scope('mobilenetv3'):
        input_data = tf.reshape(input_data, [-1, 416, 416, 3]) # print layer's shape

        input_data = conv2d_block(input_data, 16, 3, 2, trainable, name='conv1_1', h_swish=True)  # size/2

        input_data = mnv3_block(input_data, 3, 16, 16, 1, trainable, name='bneck2_1', h_swish=False, ratio=reduction_ratio, se=False)

        input_data = mnv3_block(input_data, 3, 64, 24, 2, trainable, name='bneck3_1', h_swish=False, ratio=reduction_ratio, se=False)  # size/4
        input_data = mnv3_block(input_data, 3, 72, 24, 1, trainable, name='bneck3_2', h_swish=False, ratio=reduction_ratio, se=False)

        input_data = mnv3_block(input_data, 5, 72, 40, 2, trainable, name='bneck4_1', h_swish=False, ratio=reduction_ratio, se=True)  # size/8
        input_data = mnv3_block(input_data, 5, 120, 40, 1, trainable, name='bneck4_2', h_swish=False, ratio=reduction_ratio, se=True)
        input_data = mnv3_block(input_data, 5, 120, 40, 1, trainable, name='bneck4_3', h_swish=False, ratio=reduction_ratio, se=True)

        route_1 = input_data

        input_data = mnv3_block(input_data, 3, 240, 80, 2, trainable, name='bneck5_1', h_swish=True, ratio=reduction_ratio, se=False) # size/16
        input_data = mnv3_block(input_data, 3, 200, 80, 1, trainable, name='bneck5_2', h_swish=True, ratio=reduction_ratio, se=False)
        input_data = mnv3_block(input_data, 3, 184, 80, 1, trainable, name='bneck5_3', h_swish=True, ratio=reduction_ratio, se=False)
        input_data = mnv3_block(input_data, 3, 184, 80, 1, trainable, name='bneck5_4', h_swish=True, ratio=reduction_ratio, se=False)

        input_data = mnv3_block(input_data, 3, 480, 112, 1, trainable, name='bneck6_1', h_swish=True, ratio=reduction_ratio, se=True)
        input_data = mnv3_block(input_data, 3, 672, 112, 1, trainable, name='bneck6_2', h_swish=True, ratio=reduction_ratio, se=True)

        route_2 = input_data

        input_data = mnv3_block(input_data, 5, 672, 160, 2, trainable, name='bneck7_1', h_swish=True, ratio=reduction_ratio, se=True) # size/32
        input_data = mnv3_block(input_data, 5, 960, 160, 1, trainable, name='bneck7_2', h_swish=True, ratio=reduction_ratio, se=True)
        input_data = mnv3_block(input_data, 5, 960, 160, 1, trainable, name='bneck7_3', h_swish=True, ratio=reduction_ratio, se=True)

        return route_1, route_2, input_data


def mobilenetv3_small(input_data, trainable):
    reduction_ratio = 4
    with tf.variable_scope('mobilenetv3_small'):
        input_data = tf.reshape(input_data, [-1, 416, 416, 3]) # print layer's shape

        input_data = conv2d_block(input_data, 16, 3, 2, trainable, name='conv1_1', h_swish=True)  # size/2

        input_data = mnv3_block(input_data, 3, 16, 16, 2, trainable, name='bneck2_1', h_swish=False, ratio=reduction_ratio, se=True) # size/4

        input_data = mnv3_block(input_data, 3, 72, 24, 2, trainable, name='bneck3_1', h_swish=False, ratio=reduction_ratio, se=False)  # size/8
        input_data = mnv3_block(input_data, 3, 88, 24, 1, trainable, name='bneck3_2', h_swish=False, ratio=reduction_ratio, se=False)

        route_1 = input_data

        input_data = mnv3_block(input_data, 5, 96, 40, 2, trainable, name='bneck4_1', h_swish=True, ratio=reduction_ratio, se=True)  # size/16
        input_data = mnv3_block(input_data, 5, 240, 40, 1, trainable, name='bneck4_2', h_swish=True, ratio=reduction_ratio, se=True)
        input_data = mnv3_block(input_data, 5, 240, 40, 1, trainable, name='bneck4_3', h_swish=True, ratio=reduction_ratio, se=True)

        input_data = mnv3_block(input_data, 5, 120, 48, 1, trainable, name='bneck5_1', h_swish=True, ratio=reduction_ratio, se=True)
        input_data = mnv3_block(input_data, 5, 144, 48, 1, trainable, name='bneck5_2', h_swish=True, ratio=reduction_ratio, se=True)

        route_2 = input_data

        input_data = mnv3_block(input_data, 5, 288, 96, 2, trainable, name='bneck6_1', h_swish=True, ratio=reduction_ratio, se=True) # size/32
        input_data = mnv3_block(input_data, 5, 576, 96, 1, trainable, name='bneck6_2', h_swish=True, ratio=reduction_ratio, se=True)
        input_data = mnv3_block(input_data, 5, 576, 96, 1, trainable, name='bneck6_3', h_swish=True, ratio=reduction_ratio, se=True)

        return route_1, route_2, input_data
