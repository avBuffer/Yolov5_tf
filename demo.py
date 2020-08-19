#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image


if __name__ == '__main__':
    pb_file = "./checkpoint/yolov4.pb"
    video_path = "./data/images/road.mp4"
    # video_path = 0

    num_classes = 80
    input_size = 416
    score_thresh = 0.3

    iou_type = 'diou'  #yolov4:diou, else giou
    iou_thresh = 0.45

    graph = tf.Graph()
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    with tf.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)
        while True:
            return_value, frame = vid.read()
            if return_value:
                image = Image.fromarray(frame)
                frame_size = frame.shape[:2]
                image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]
                prev_time = time.time()

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                            feed_dict={return_tensors[0]: image_data})

                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)), np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, score_thresh)
                bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')
                image = utils.draw_bbox(frame, bboxes)

                curr_time = time.time()
                exec_time = curr_time - prev_time
                
                result = np.asarray(image)
                info = "time: %.2f ms" % (1000 * exec_time)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            else:
                print('Finish processing!')
                raise ValueError("No image!")
                break
