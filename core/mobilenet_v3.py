#coding:utf-8
from core.mnv3_layers import *


def mobilenetv3(inputs, num_classes, is_train=True):
    reduction_ratio = 4
    with tf.variable_scope('mobilenetv3'):
        net = conv2d_block(inputs, 16, 3, 2, is_train, name='conv1_1', h_swish=True)  # size/2

        net = mnv3_block(net, 3, 16, 16, 1, is_train, name='bneck2_1', h_swish=False, ratio=reduction_ratio, se=False)

        net = mnv3_block(net, 3, 64, 24, 2, is_train, name='bneck3_1', h_swish=False, ratio=reduction_ratio, se=False)  # size/4
        net = mnv3_block(net, 3, 72, 24, 1, is_train, name='bneck3_2', h_swish=False, ratio=reduction_ratio, se=False)

        net = mnv3_block(net, 5, 72, 40, 2, is_train, name='bneck4_1', h_swish=False, ratio=reduction_ratio, se=True)  # size/8
        net = mnv3_block(net, 5, 120, 40, 1, is_train, name='bneck4_2', h_swish=False, ratio=reduction_ratio, se=True)
        net = mnv3_block(net, 5, 120, 40, 1, is_train, name='bneck4_3', h_swish=False, ratio=reduction_ratio, se=True)

        net = mnv3_block(net, 3, 240, 80, 2, is_train, name='bneck5_1', h_swish=True, ratio=reduction_ratio, se=False) # size/16
        net = mnv3_block(net, 3, 200, 80, 1, is_train, name='bneck5_2', h_swish=True, ratio=reduction_ratio, se=False)
        net = mnv3_block(net, 3, 184, 80, 1, is_train, name='bneck5_3', h_swish=True, ratio=reduction_ratio, se=False)
        net = mnv3_block(net, 3, 184, 80, 1, is_train, name='bneck5_4', h_swish=True, ratio=reduction_ratio, se=False)

        net = mnv3_block(net, 3, 480, 112, 1, is_train, name='bneck6_1', h_swish=True, ratio=reduction_ratio, se=True)
        net = mnv3_block(net, 3, 672, 112, 1, is_train, name='bneck6_2', h_swish=True, ratio=reduction_ratio, se=True)

        net = mnv3_block(net, 5, 672, 160, 2, is_train, name='bneck7_2', h_swish=True, ratio=reduction_ratio, se=True) # size/32
        net = mnv3_block(net, 5, 960, 160, 1, is_train, name='bneck7_3', h_swish=True, ratio=reduction_ratio, se=True)
        net = mnv3_block(net, 5, 960, 160, 1, is_train, name='bneck7_1', h_swish=True, ratio=reduction_ratio, se=True)

        net = conv2d_hs(net, 960, is_train, name='conv8_1')
        net = global_avg(net, 7)

        net = conv2d_NBN_hs(net, 1280, name='conv2d_NBN', bias=True)
        net = conv_1x1(net, num_classes, name='logits', bias=True)

        logits = flatten(net)
        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred


def mobilenetv3_small(inputs, num_classes, is_train=True):
    reduction_ratio = 4
    with tf.variable_scope('mobilenetv3_small'):
        net = conv2d_block(inputs, 16, 3, 2, is_train, name='conv1_1', h_swish=True)  # size/2

        net = mnv3_block(net, 3, 16, 16, 2, is_train, name='bneck2_1', h_swish=False, ratio=reduction_ratio, se=True) # size/4

        net = mnv3_block(net, 3, 72, 24, 2, is_train, name='bneck3_1', h_swish=False, ratio=reduction_ratio, se=False)  # size/8
        net = mnv3_block(net, 3, 88, 24, 1, is_train, name='bneck3_2', h_swish=False, ratio=reduction_ratio, se=False)

        net = mnv3_block(net, 5, 96, 40, 1, is_train, name='bneck4_1', h_swish=True, ratio=reduction_ratio, se=True)  # size/16
        net = mnv3_block(net, 5, 240, 40, 1, is_train, name='bneck4_2', h_swish=True, ratio=reduction_ratio, se=True)
        net = mnv3_block(net, 5, 240, 40, 1, is_train, name='bneck4_3', h_swish=True, ratio=reduction_ratio, se=True)

        net = mnv3_block(net, 5, 120, 48, 1, is_train, name='bneck5_1', h_swish=True, ratio=reduction_ratio, se=True)
        net = mnv3_block(net, 5, 144, 48, 1, is_train, name='bneck5_2', h_swish=True, ratio=reduction_ratio, se=True)

        net = mnv3_block(net, 5, 288, 96, 2, is_train, name='bneck6_1', h_swish=True, ratio=reduction_ratio, se=True) # size/32
        net = mnv3_block(net, 5, 576, 96, 1, is_train, name='bneck6_2', h_swish=True, ratio=reduction_ratio, se=True)
        net = mnv3_block(net, 5, 576, 96, 1, is_train, name='bneck6_3', h_swish=True, ratio=reduction_ratio, se=True)

        net = conv2d_hs(net, 576, is_train, name='conv7_1', se=True)  #SE
        net = global_avg(net, 7)

        net = conv2d_NBN_hs(net, 1280, name='conv2d_NBN', bias=True)
        net = conv_1x1(net, num_classes, name='logits', bias=True)

        logits = flatten(net)
        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred
