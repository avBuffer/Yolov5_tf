#! /usr/bin/env python
# coding=utf-8
import core.common as common
import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def darknet53(input_data, trainable):
    with tf.variable_scope('darknet'):
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
        #input_data = tf.reshape(input_data, [-1, 608, 608, 3]) # print layer's shape

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

