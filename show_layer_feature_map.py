#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import core.utils as utils
import tensorflow as tf
from PIL import Image


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 5:
        print('usage: python show_layer_feature_map.py gpu_id pb_file img_file out_path')
        sys.exit()

    gpu_id = argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pb_file = argv[2]
    if not os.path.exists(pb_file):
        print('pb_file=%s not exist' % pb_file)
        sys.exit()

    img_file = argv[3]
    if not os.path.exists(img_file):
        print('img_file=%s not exist' % img_file)
        sys.exit()

    out_path = argv[4]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('show_layer_feature_map gpu_id=%s, pb_file=%s, img_file=%s, out_path=%s' % 
         (gpu_id, pb_file, img_file, out_path))
    
    input_size = 416
    img = cv2.imread(img_file)
    image_data = utils.image_preporcess(np.copy(img), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    
    graph = tf.Graph()
    return_elements = ['input/input_data:0', 'pred_sbbox/concat_2:0', 'pred_mbbox/concat_2:0', 'pred_lbbox/concat_2:0']
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        tensor_names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        conv_layer_names = []
        for idx, tensor_name in enumerate(tensor_names):
            if 'Conv2D' in tensor_name:
                conv_layer_names.append(tensor_name)
        print('conv_layer_names=', conv_layer_names)

        for idx, layer_name in enumerate(conv_layer_names):
            conv = sess.graph.get_tensor_by_name('%s:0' % layer_name)
            features = np.array(conv.eval({return_tensors[0]: image_data}))
            print('\n[%d/%d] %s' % (idx, len(conv_layer_names), layer_name), ' features.shape=', features.shape)

            out_layer_path = os.path.join(out_path, '%s-%sx%sx%s' % (layer_name.replace('/', '_'), str(features.shape[1]), 
                                                                     str(features.shape[2]), str(features.shape[3])))
            if not os.path.exists(out_layer_path):
                os.makedirs(out_layer_path)

            plt.figure(idx, figsize=(10, 10))
            for jdx in range(features.shape[3]):
                plt.matshow(features[0, :, :, jdx], cmap=plt.cm.gray, fignum=idx) #remove cmap=plt.cm.gray to show RGBA image
                plt.title('' + layer_name + '_' + str(jdx))

                out_file = os.path.join(out_layer_path, 'img_%s.jpg' % str(jdx))
                plt.savefig(out_file)
                print('idx=', idx, ' layer_name=', layer_name, ' jdx=', jdx, ' out_file=', out_file)
