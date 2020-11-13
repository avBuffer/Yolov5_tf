# -*- coding: utf-8 -*-
import os
import sys
from core.yolov3 import YOLOV3
from core.yolov4 import YOLOV4
from core.yolov5 import YOLOV5

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


if __name__ == "__main__":
    """
    argv = sys.argv
    if len(argv) < 5:
        print('usage: python freeze_ckpt_to_pb.py gpu_id net_type(yolov5/yolov4/yolov3) ckpt_file pb_file')
        sys.exit()
    """
    gpu_id = '0' #argv[1]
    net_type = 'yolov3' #argv[2]
    ckpt_file = 'ckpts/social_yolov3_test-loss=3.2020.ckpt-198' #argv[3]
    if not os.path.exists(ckpt_file + '.index'):
        print('freeze_ckpt_to_pb ckpt_file=', ckpt_file, ' not exist')
        sys.exit()

    pb_file = ckpt_file + '.pb' #argv[4]
    print('freeze_ckpt_to_pb gpu_id=%s, net_type=%s, ckpt_file=%s, pb_file=%s' % (gpu_id, net_type, ckpt_file, pb_file))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    if net_type == 'yolov3':
        model = YOLOV3(input_data, trainable=False)
    elif net_type == 'yolov4':
        model = YOLOV4(input_data, trainable=False)
    elif net_type == 'yolov5':
        model = YOLOV5(input_data, trainable=False)
    else:
        print('net_type=', net_type, ' error')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_node_names)
    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())
