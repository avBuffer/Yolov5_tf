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

    pb_file = 'ckpts/MosEggs_yolov3_loss=33.0915.ckpt-8.pb' #argv[2]
    if not os.path.exists(pb_file):
        print('pb_file=%s not exist' % pb_file)
        sys.exit()

    img_path_file = 'D:/datasets/MosEggs/test' #argv[3]
    if not os.path.exists(img_path_file):
        print('img_path_file=%s not exist' % img_path_file)
        sys.exit()

    out_path = 'D:/datasets/MosEggs/out' #argv[4]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('test gpu_id=%s, pb_file=%s, img_file=%s, out_path=%s' % (gpu_id, pb_file, img_path_file, out_path))

    num_classes = 1
    input_size = 512
    score_thresh = 0.3

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
            # 开始切图 cut
            h_step = img.shape[0] // input_size
            w_step = img.shape[1] // input_size
            h_rest = -(img.shape[0] - input_size * h_step)
            w_rest = -(img.shape[1] - input_size * w_step)
            img_list = []

            # 循环切图
            for h in range(h_step):
                for w in range(w_step):
                    # 划窗采样
                    im = img[(h * input_size) : (h * input_size + input_size), (w * input_size) : (w * input_size + input_size), :]
                    img_list.append(im)
                img_list.append(img[(h * input_size) : (h * input_size + input_size), -input_size:, :])

            for w in range(w_step - 1):
                img_list.append(img[-input_size:, (w * input_size) : (w * input_size + input_size), :])
            img_list.append(img[-input_size:, -input_size:, :])

            predict_img_list = []
            for i, ims in enumerate(img_list):
                img_size = ims.shape[:2]
                image_data = utils.image_preporcess(np.copy(ims), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                                                              feed_dict={return_tensors[0]: image_data})
                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
                bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')

                out_img = ims
                if len(bboxes) > 0:
                    image = utils.draw_bbox(ims, bboxes)
                    # image = Image.fromarray(image)
                    # image.show()
                    out_img = np.asarray(image)
                    score = bboxes[0][4]

                    file_path, file_name = os.path.split(img_path_file)
                    file, postfix = os.path.splitext(file_name)
                    # out_file = os.path.join(out_path, file + '_%.6f' % (score) + postfix)
                    # cv2.imwrite(out_file, out_img)
                    # print('idx=', idx, 'in_img_file=', in_img_file, 'out_file=', out_file)
                predict_img_list.append(out_img)

            # 将预测后的图像块再拼接起来
            count_temp = 0
            result_img = img.copy()
            for h in range(h_step):
                for w in range(w_step):
                    result_img[h * input_size: (h + 1) * input_size, w * input_size: (w + 1) * input_size] = predict_img_list[count_temp]
                    count_temp += 1
                result_img[h * input_size: (h + 1) * input_size, w_rest:] = predict_img_list[count_temp][:, w_rest:]
                count_temp += 1

            for w in range(w_step - 1):
                result_img[h_rest:, (w * input_size) : (w * input_size + input_size)] = predict_img_list[count_temp][h_rest:, :]
                count_temp += 1

            result_img[h_rest:, w_rest:] = predict_img_list[count_temp][h_rest:, w_rest:]
            out_file = os.path.join(out_path, img_path_file.replace('.jpg', '_result.jpg'))
            cv2.imwrite(out_file, result_img)
            print('in_img_file=', img_path_file, 'out_file=', out_file)

        elif os.path.isdir(img_path_file):
            def detect_img(fim, start_h, start_w, fpredict_bboxes):
                img_size = fim.shape[:2]
                image_data = utils.image_preporcess(np.copy(fim), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                                                              feed_dict={return_tensors[0]: image_data})
                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
                bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')

                for i, bbox in enumerate(bboxes):
                    # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates
                    coor = np.array(bbox[:4], dtype=np.int32)
                    bbox[0] = int(start_w + coor[0])
                    bbox[1] = int(start_h + coor[1])
                    bbox[2] = int(start_w + coor[2])
                    bbox[3] = int(start_h + coor[3])
                    fpredict_bboxes.append(bbox)

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

                # 开始切图 cut
                h_step = img.shape[0] // input_size
                w_step = img.shape[1] // input_size
                h_rest = -(img.shape[0] - input_size * h_step)
                w_rest = -(img.shape[1] - input_size * w_step)
                predict_bboxes = []
                # 循环切图
                for h in range(h_step):
                    for w in range(w_step):
                        # 划窗采样
                        im = img[(h * input_size) : (h * input_size + input_size), (w * input_size) : (w * input_size + input_size), :]
                        detect_img(im, h * input_size, w * input_size,  predict_bboxes)

                    # for w_rest
                    im = img[(h * input_size) : (h * input_size + input_size), -input_size:, :]
                    detect_img(im, h * input_size, w_rest, predict_bboxes)

                # for h_rest
                for w in range(w_step - 1):
                    im = img[-input_size:, (w * input_size) : (w * input_size + input_size), :]
                    detect_img(im, h_rest, w * input_size, predict_bboxes)

                # for h_rest and w_rest
                im = img[-input_size:, -input_size:, :]
                detect_img(im, h_rest, w_rest, predict_bboxes)

                if len(predict_bboxes) > 0:
                    image = utils.draw_bbox(img, predict_bboxes)
                    out_img = np.asarray(image)
                    score = predict_bboxes[0][4]

                    file_path, file_name = os.path.split(in_img_file)
                    file, postfix = os.path.splitext(file_name)
                    out_file = os.path.join(out_path, file + '_%d_%.6f' % (input_size, score) + postfix)
                    cv2.imwrite(out_file, out_img)
                    print('idx=', idx, 'in_img_file=', in_img_file, 'out_file=', out_file, 'predict_bboxes.len=', len(predict_bboxes))
                #break

        elif os.path.isdir(img_path_file) and False:
            img_files = os.listdir(img_path_file)
            for idx, img_file in enumerate(img_files):
                in_img_file = os.path.join(img_path_file, img_file)
                # print('idx=', idx, 'in_img_file=', in_img_file)
                if not os.path.exists(in_img_file):
                    print('idx=', idx, 'in_img_file=', in_img_file, ' not exist')
                    continue

                img = cv2.imread(in_img_file)
                if img is None:
                    print('idx=', idx, 'in_img_file=', in_img_file, ' read error')
                    continue

                # 开始切图 cut
                h_step = img.shape[0] // input_size
                w_step = img.shape[1] // input_size
                h_rest = -(img.shape[0] - input_size * h_step)
                w_rest = -(img.shape[1] - input_size * w_step)
                img_list = []

                # 循环切图
                for h in range(h_step):
                    for w in range(w_step):
                        # 划窗采样
                        im = img[(h * input_size) : (h * input_size + input_size), (w * input_size) : (w * input_size + input_size), :]
                        img_list.append(im)
                    img_list.append(img[(h * input_size):(h * input_size + input_size), -input_size:, :])

                for w in range(w_step - 1):
                    img_list.append(img[-input_size:, (w * input_size):(w * input_size + input_size), :])
                img_list.append(img[-input_size:, -input_size:, :])

                predict_img_list = []
                predict_bboxes_list = []
                for i, ims in enumerate(img_list):
                    img_size = ims.shape[:2]
                    image_data = utils.image_preporcess(np.copy(ims), [input_size, input_size])
                    image_data = image_data[np.newaxis, ...]

                    pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]],
                                                                   feed_dict={return_tensors[0]: image_data})
                    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                    bboxes = utils.postprocess_boxes(pred_bbox, img_size, input_size, score_thresh)
                    bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')

                    out_img = ims
                    if len(bboxes) > 0:
                        image = utils.draw_bbox(ims, bboxes)
                        out_img = np.asarray(image)
                        for i, bbox in enumerate(bboxes):
                            predict_bboxes_list.append(bbox)

                        # score = bboxes[0][4]
                        # file_path, file_name = os.path.split(in_img_file)
                        # file, postfix = os.path.splitext(file_name)
                        # out_file = os.path.join(out_path, file + '_%.6f' % (score) + postfix)
                        # cv2.imwrite(out_file, out_img)
                        # print('idx=', idx, 'in_img_file=', in_img_file, 'out_file=', out_file)
                    predict_img_list.append(out_img)

                # 将预测后的图像块再拼接起来
                count_temp = 0
                result_img = img.copy()
                for h in range(h_step):
                    for w in range(w_step):
                        result_img[h * input_size: (h + 1) * input_size, w * input_size: (w + 1) * input_size] = predict_img_list[count_temp]
                        count_temp += 1
                    result_img[h * input_size: (h + 1) * input_size, w_rest:] = predict_img_list[count_temp][:, w_rest:]
                    count_temp += 1

                if h_rest != 0:
                    for w in range(w_step - 1):
                        result_img[h_rest:, (w * input_size): (w * input_size + input_size)] = predict_img_list[count_temp][h_rest:,:]
                        count_temp += 1
                    result_img[h_rest:, w_rest:] = predict_img_list[count_temp][h_rest:, w_rest:]

                out_file = os.path.join(out_path, img_file.replace('.jpg', '_%d.jpg' % input_size))
                cv2.imwrite(out_file, result_img)
                print('idx=', idx, 'in_img_file=', in_img_file, 'out_file=', out_file, 'predict_bboxes_list.len=', len(predict_bboxes_list))
                break

        else:
            print('img_path_file=%s is error' % img_path_file)
