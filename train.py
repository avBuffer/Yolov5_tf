#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
import numpy as np
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3_tiny import YOLOV3Tiny
from core.yolov3 import YOLOV3
from core.yolov4 import YOLOV4
from core.yolov5 import YOLOV5
from core.config import cfg

import tensorflow
print('tensorflow.version=', tensorflow.__version__)
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


class YoloTrain(object):
    def __init__(self, net_type):
        self.net_type = net_type
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        
        self.ckpt_path = cfg.TRAIN.CKPT_PATH        
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150

        self.log_path = ('log/%s' % net_type)
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
            #os.removedirs(self.log_path)
        os.makedirs(self.log_path)

        self.trainset = Dataset('train', self.net_type)
        self.testset = Dataset('test', self.net_type)
        self.steps_per_period = len(self.trainset)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        with tf.name_scope('input'):
            if net_type == 'tiny':
                self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
                self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
                self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')

                self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
                self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
                self.trainable = tf.placeholder(dtype=tf.bool, name='training')

            else:                
                self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
                self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
                self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
                self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')

                self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
                self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
                self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
                self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            if self.net_type == 'tiny':
                self.model = YOLOV3Tiny(self.input_data, self.trainable)
                self.net_var = tf.global_variables()
                self.iou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_mbbox, self.label_lbbox,
                                                                                        self.true_mbboxes, self.true_lbboxes)
                self.loss = self.iou_loss + self.conf_loss + self.prob_loss

            elif self.net_type == 'yolov3':
                self.model = YOLOV3(self.input_data, self.trainable)
                self.net_var = tf.global_variables()
                self.iou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_sbbox, self.label_mbbox, self.label_lbbox,
                                                                                        self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
                self.loss = self.iou_loss + self.conf_loss + self.prob_loss
            
            elif self.net_type == 'yolov4' or self.net_type == 'yolov5':
                iou_use = 1  # (0, 1, 2) ==> (giou_loss, diou_loss, ciou_loss)
                focal_use = False  # (False, True) ==> (normal, focal_loss)
                label_smoothing = 0

                if self.net_type == 'yolov4':
                    self.model = YOLOV4(self.input_data, self.trainable)
                else:
                    self.model = YOLOV5(self.input_data, self.trainable)

                self.net_var = tf.global_variables()
                self.iou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_sbbox, self.label_mbbox, self.label_lbbox,
                                                                                        self.true_sbboxes, self.true_mbboxes, self.true_lbboxes,
                                                                                        iou_use, focal_use, label_smoothing)
                self.loss = self.iou_loss + self.conf_loss + self.prob_loss
                # self.loss = tf.Print(self.loss, [self.iou_loss, self.conf_loss, self.prob_loss], message='loss: ')
            else:
                print('self.net_type=%s error' % self.net_type)

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period, dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                       dtype=tf.float64, name='train_steps')
            
            self.learn_rate = tf.cond(pred=self.global_step < warmup_steps, true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                                      false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * \
                                              (1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope('define_weight_decay'):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('define_first_stage_train'):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if net_type == 'tiny':
                    bboxes = ['conv_mbbox', 'conv_lbbox']
                else:
                    bboxes = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']
                
                if var_name_mess[0] in bboxes:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope('define_second_stage_train'):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        with tf.name_scope('summary'):
            tf.summary.scalar('learn_rate', self.learn_rate)
            tf.summary.scalar('iou_loss', self.iou_loss)
            tf.summary.scalar('conf_loss', self.conf_loss)
            tf.summary.scalar('prob_loss', self.prob_loss)
            tf.summary.scalar('total_loss', self.loss)

            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLO-%s from scratch ...' % self.net_type)
            self.first_stage_epochs = 0

        saving = 0.0
        for epoch in range(1, (1 + self.first_stage_epochs + self.second_stage_epochs)):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                if net_type == 'tiny':
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step], 
                        feed_dict={self.input_data: train_data[0],
                                   self.label_mbbox: train_data[1], self.label_lbbox: train_data[2],
                                   self.true_mbboxes: train_data[3], self.true_lbboxes: train_data[4], 
                                   self.trainable: True,})
                else:
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step], 
                        feed_dict={self.input_data: train_data[0],
                                   self.label_sbbox: train_data[1], self.label_mbbox: train_data[2], self.label_lbbox: train_data[3],
                                   self.true_sbboxes: train_data[4], self.true_mbboxes: train_data[5], self.true_lbboxes: train_data[6], 
                                   self.trainable: True,}) 

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description('train loss: %.2f' %train_step_loss)

            for test_data in self.testset:
                if net_type == 'tiny':
                    test_step_loss = self.sess.run(self.loss, 
                        feed_dict={self.input_data: test_data[0], 
                                   self.label_mbbox: test_data[1], self.label_lbbox: test_data[2], 
                                   self.true_mbboxes: test_data[3], self.true_lbboxes: test_data[4], 
                                   self.trainable: False,})
                else:
                    test_step_loss = self.sess.run(self.loss, 
                        feed_dict={self.input_data: test_data[0], 
                                   self.label_sbbox: test_data[1], self.label_mbbox: test_data[2], self.label_lbbox: test_data[3], 
                                   self.true_sbboxes: test_data[4], self.true_mbboxes: test_data[5], self.true_lbboxes: test_data[6], 
                                   self.trainable: False,})
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            train_epoch_loss = np.mean(train_epoch_loss)
            
            ckpt_file = os.path.join(self.ckpt_path, 'social_%s_test-loss=%.4f.ckpt' % (self.net_type, test_epoch_loss))
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if saving == 0.0:
                saving = train_epoch_loss
                print('=> Epoch: %2d Time: %s Train loss: %.2f' % (epoch, log_time, train_epoch_loss))
            
            elif saving > train_epoch_loss:
                print('=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ...' % 
                     (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                self.saver.save(self.sess, ckpt_file, global_step=epoch)
                saving = train_epoch_loss
            
            else:
                print('=> Epoch: %2d Time: %s Train loss: %.2f' % (epoch, log_time, train_epoch_loss))


if __name__ == '__main__':
    """
    argv = sys.argv
    if len(argv) < 3:
        print('usage: python train.py gpu_id net_type(yolov5/yolov4/yolov3/tiny)')
        sys.exit()

    """
    gpu_id = 0 #argv[1]
    net_type = 'yolov3' #argv[2]
    print('train gpu_id=%s, net_type=%s' % (gpu_id, net_type))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    YoloTrain(net_type).train()
