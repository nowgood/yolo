# -*-coding:utf-8-*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from utils.dataset_utils import recursive_parse_xml_to_dict
from utils.label_map_util import get_label_map_dict
from dataset_tools.create_pascal_tf_record import dict_to_tf_example
import os
from net import yolo_net


flags = tf.app.flags

flags.DEFINE_string("tfrecord_path", "", "path to tfrecord file")

FLAGS = flags.FLAGS


def tfrecord2example():
    return {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/key/sha256': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        # 由于一张图片中的 object 的个数时变化的, 所以要用 tf.VarLenFeature(),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':  tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/difficult': tf.VarLenFeature(tf.int64),
        'image/object/truncated': tf.VarLenFeature(tf.int64),
        'image/object/view': tf.VarLenFeature(tf.string),
    }


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(cwd, "../test_image/")
    label_map_path = os.path.join(cwd, "../data/pascal_label_map.pbtxt")

    tf_reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([FLAGS.tfrecord_path])
    _, image_raw = tf_reader.read(file_queue)
    example = tfrecord2example()
    features = tf.parse_single_example(image_raw, features=example)

    label = features['image/object/class/label']
    image = features['image/encoded']
    xmin = features['image/object/bbox/xmin']
    ymin = features['image/object/bbox/ymin']
    xmax = features['image/object/bbox/xmax']
    ymax = features['image/object/bbox/ymax']
    height = features['image/height']
    width = features['image/width']

    with tf.Session() as sess:
        label, image, height, width, xmin, ymin, xmax, ymax = \
            sess.run(label, image, height, width, xmin, ymin, xmax, ymax)
        net = yolo_net(image)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image = tf.image.decode_jpeg(image, channels=3)

        coord.request_stop()
        coord.join(threads)



