#coding:utf-8
import numpy as np
import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

weight_decay = 1e-5


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.name_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
        return h_swish


def batch_norm(x, momentum=0.997, epsilon=1e-3, train=True, name='bn'):
    return tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, scale=True,
                                         center=True, training=train, name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.09, name='conv2d', bias=False):
    with tf.variable_scope(name):
        if tensorflow.__version__.startswith('1.'):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
        else:
            w = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=[k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def conv2d_block(input, out_dim, k, s, is_train, name, h_swish=False):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')

        if h_swish == True:
            net = hard_swish(net)
        else:
            net = relu6(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.09, name=name, bias=bias)


def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out = conv_1x1(input, output_dim, bias=bias, name='pwb')
        out = batch_norm(out, train=is_train, name='bn')
        out = relu6(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=[1, 1, 1, 1],
               padding='SAME', stddev=0.09, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        if tensorflow.__version__.startswith('1.'):
            w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
        else:
            w = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=[k_h, k_w, in_channel, channel_multiplier],
                                initializer=tf.random_normal_initializer(stddev=0.01))

        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)

        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def Fully_connected(x, units, layer_name='fully_connected'):
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)


def global_avg(x, s=1):
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], s)
        return net


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.name_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
        return h_sigmoid


def conv2d_hs(input, output_dim, is_train, name, bias=False, se=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out = conv_1x1(input, output_dim, bias=bias, name='pwb')
        out = batch_norm(out, train=is_train, name='bn')
        out = hard_swish(out)

        # squeeze and excitation
        if se:
            channel = int(np.shape(out)[-1])
            out = squeeze_excitation_layer(out,out_dim=channel, ratio=4, layer_name='se_block')
        return out


def conv2d_NBN_hs(input, output_dim, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out = conv_1x1(input, output_dim, bias=bias, name='pwb')
        out = hard_swish(out)
        return out


def squeeze_excitation_layer(input, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = global_avg(input)
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_excitation1')
        excitation = relu6(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_excitation2')
        excitation = hard_sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale


def mnv3_block(input, k_s, expansion_ratio, output_dim, stride, is_train, name,
               bias=True, shortcut=True, h_swish=False, ratio=16, se=False):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim = expansion_ratio #round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')

        if h_swish:
            net = hard_swish(net)
        else:
            net = relu6(net)

        # dw
        net = dwise_conv(net, k_w=k_s, k_h=k_s, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        if h_swish:
            net = hard_swish(net)
        else:
            net = relu6(net)

        # squeeze and excitation
        if se:
            channel = int(np.shape(net)[-1])
            net = squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(input.get_shape().as_list()[-1])
            net_dim = int(net.get_shape().as_list()[-1])

            if in_dim == net_dim:
                net += input
                net = tf.identity(net, name='output')
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)
