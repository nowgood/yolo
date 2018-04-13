# -*-coding:utf-8-*-

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib import layers as layers_lib

BOX_PER_CELL = 2
NUM_CELL = 7
NUM_CLASS = 20


def yolonet(images, is_training=True):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        output_depth = NUM_CLASS + 5 * BOX_PER_CELL
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            bottleneck, _ = resnet_v2.resnet_v2_50(images, global_pool=False,
                                                   is_training=is_training)
            with arg_scope([layers.batch_norm], is_training=is_training):
                net = bottleneck
                net = layers_lib.conv2d(net, 512, [1, 1],
                                        normalizer_fn=layers.batch_norm, scope='yolo_layer1')
                net = layers_lib.conv2d(net, 512, [3, 3],
                                        normalizer_fn=layers.batch_norm, scope='yolo_layer2')
                net = layers_lib.conv2d(net, 512, [3, 3],
                                        normalizer_fn=layers.batch_norm, scope='yolo_layer3')
                net = layers_lib.conv2d(net, output_depth, [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope='yolo_output')

    return net



