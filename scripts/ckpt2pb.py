# -*- coding: utf-8 -*-
import tensorflow as tf
from core.yolov3 import YOLOV3
from core.yolov4 import YOLOV4


def freeze_graph(pb_file, ckpt_file, output_node_names):
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    model = YOLOV4(input_data, trainable=False)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
        input_graph_def=sess.graph.as_graph_def(), output_node_names=output_node_names)

    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())


if __name__ == "__main__":
    pb_file = "./checkpoint/yolov4.pb"
    ckpt_file = "./checkpoint/yolov4"
    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
    freeze_graph(pb_file, ckpt_file, output_node_names)
