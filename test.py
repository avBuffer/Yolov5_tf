#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import core.utils as utils
from PIL import Image

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


if __name__ == '__main__':
    """
    argv = sys.argv
    if len(argv) < 5:
        print('usage: python test.py gpu_id pb_file img_path_file out_path')
        sys.exit()
    """
    gpu_id = '0' #argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pb_file = 'ckpts/social_yolov3_test-loss=3.2020.ckpt-198.pb' #argv[2]
    if not os.path.exists(pb_file):
        print('pb_file=%s not exist' % pb_file)
        sys.exit()

    img_path_file = 'D:/datasets/Social/test' #argv[3]
    if not os.path.exists(img_path_file):
        print('img_path_file=%s not exist' % img_path_file)
        sys.exit()

    out_path = 'D:/datasets/Social/out' #argv[4]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('test gpu_id=%s, pb_file=%s, img_file=%s, out_path=%s' % (gpu_id, pb_file, img_path_file, out_path))

    num_classes = 1
    input_size = 416
    score_thresh = 0.6

    iou_type = 'iou' #yolov4:diou, else giou
    iou_thresh = 0.3

    graph = tf.Graph()
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        if os.path.isfile(img_path_file):
            img = cv2.imread(img_path_file)            
            img_size = img.shape[:2]
            image_data = utils.image_preporcess(np.copy(img), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
            bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')
            
            if len(bboxes) > 0:
                image = utils.draw_bbox(img, bboxes) 
                #image = Image.fromarray(image)
                #image.show()
                out_img = np.asarray(image)
                score = bboxes[0][4]

                file_path, file_name = os.path.split(img_path_file)
                file, postfix = os.path.splitext(file_name)
                out_file = os.path.join(out_path, file + '_%.6f' % (score) + postfix)

                cv2.imwrite(out_file, out_img)
                print('img_path_file=', img_path_file, 'out_file=', out_file)

        elif os.path.isdir(img_path_file):
            img_files = os.listdir(img_path_file)
            for idx, img_file in enumerate(img_files):
                in_img_file = os.path.join(img_path_file, img_file)
                #print('idx=', idx, 'in_img_file=', in_img_file)
                if not os.path.exists(in_img_file):
                    print('idx=', idx, 'in_img_file=', in_img_file, ' not exist')
                    continue
            
                img = cv2.imread(in_img_file)
                if img is None:
                    print('idx=', idx, 'in_img_file=', in_img_file, ' read error')
                    continue

                img_size = img.shape[:2]
                image_data = utils.image_preporcess(np.copy(img), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                                                              feed_dict={return_tensors[0]: image_data})

                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
                bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')

                if len(bboxes) > 0:
                    image = utils.draw_bbox(img, bboxes)
                    #image = Image.fromarray(image)
                    #image.show()
                    out_img = np.asarray(image)
                    score = bboxes[0][4]

                    file_path, file_name = os.path.split(in_img_file)
                    file, postfix = os.path.splitext(file_name)
                    out_file = os.path.join(out_path, file + '_%.6f' % (score) + postfix)

                    cv2.imwrite(out_file, out_img)
                    print('idx=', idx, 'in_img_file=', in_img_file, 'out_file=', out_file)
        else:
            print('img_path_file=%s is error' % img_path_file)
