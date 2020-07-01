#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 5:
        print('usage: python test.py gpu_id pb_file img_path_file out_path')
        sys.exit()

    gpu_id = argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pb_file = argv[2]
    if not os.path.exist(pb_file):
        print('pb_file=%s not exist' % pb_file)
        sys.exit()

    img_path_file = argv[3]
    if not os.path.exist(img_path_file):
        print('img_path_file=%s not exist' % img_path_file)
        sys.exit()

    out_path = argv[4]
    if not os.path.exist(out_path):
        os.makedirs(out_path)
    print('test gpu_id=%s, pb_file=%s, img_file=%s, out_path=%s' % (gpu_id, pb_file, img_path_file, out_path))
    
       
    num_classes = 80
    input_size = 416
    score_thresh = 0.3
    iou_thresh = 0.45

    graph = tf.Graph()
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        if os.path.isfile(img_path_file):
            img = cv2.imread(img_path_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_size = img.shape[:2]
            image_data = utils.image_preporcess(np.copy(img), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
            bboxes = utils.nms(bboxes, iou_thresh, method='nms')
            
            if len(bboxes) > 0:
	            image = utils.draw_bbox(img, bboxes) 
	            #image = Image.fromarray(image)
	            #image.show()
	            out_img = np.asarray(image)
	            score = bboxes[0][4]

	            file_name, file_path = os.path.split(img_path_file)
	            file, postfix = os.path.splitext(file_name)
	            out_file = os.path.join(out_path, str(score) + '_' + file_name)
	            cv2.imwrite(out_file, out_img)

	    elif os.path.isdir(img_path_file):
	    	img_files = os.listdir(img_path_file)
	    	for idx, img_file in enumerate(img_files):
	    		in_img_file = os.path.join(img_path_file, img_file)
	    		if not.os.path.exist(in_img_file):
	    			continue
            
            img = cv2.imread(img_files)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_size = img.shape[:2]
            image_data = utils.image_preporcess(np.copy(img), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
            bboxes = utils.nms(bboxes, iou_thresh, method='nms')
            
            if len(bboxes) > 0:
	            image = utils.draw_bbox(img, bboxes) 
	            #image = Image.fromarray(image)
	            #image.show()
	            out_img = np.asarray(image)
	            score = bboxes[0][4]

	            file_name, file_path = os.path.split(img_path_file)
	            file, postfix = os.path.splitext(file_name)
	            out_file = os.path.join(out_path, str(score) + '_' + file_name)
	            cv2.imwrite(out_file, out_img)
        
        else:
        	print('img_path_file=%s is error' % img_path_file)
