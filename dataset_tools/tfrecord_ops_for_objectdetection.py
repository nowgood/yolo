# -*-coding:utf-8-*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from utils.dataset_utils import recursive_parse_xml_to_dict
from utils.label_map_util import get_label_map_dict
from dataset_tools.create_pascal_tf_record import dict_to_tf_example
import os


def image2tfrecord(image_dir, tfrecord_path, label_map_path ):

    with tf.gfile.GFile("../data/annotation_test.xml", 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)['annotation']
    label_map_dict = get_label_map_dict(label_map_path)

    example = dict_to_tf_example(data=data, dataset_directory=image_dir,
                                 label_map_dict=label_map_dict,
                                 image_subdirectory="")
    with tf.python_io.TFRecordWriter(tfrecord_path) as tf_writer:
        tf_writer.write(example.SerializeToString())


def tfrecord2example():
    return {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/key/sha256': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
        'image/object/class/text': tf.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.FixedLenFeature([], tf.int64),
        'image/object/difficult': tf.FixedLenFeature([], tf.int64),
        'image/object/truncated': tf.FixedLenFeature([], tf.int64),
        'image/object/view': tf.FixedLenFeature([], tf.string),
    }


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(cwd, "../test_image/")
    tfrecord_path = os.path.join(cwd, "../test_image/object_detection.tfrecords")
    label_map_path = os.path.join(cwd, "../data/pascal_label_map.pbtxt")

    print("image_dir", os.path.abspath(image_dir))

    if not os.path.exists(tfrecord_path):
        image2tfrecord(image_dir, tfrecord_path, label_map_path)

    tf_reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([tfrecord_path])
    _, image_raw = tf_reader.read(file_queue)
    example = tfrecord2example()
    features = tf.parse_single_example(image_raw, features=example)
    text = tf.cast(features['image/object/class/text'], tf.string)
    label = features['image/object/class/label']
    image = features['image/encoded']

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image = tf.image.decode_jpeg(image, channels=3)
        for _ in range(1):
            text_x, label_x, image = sess.run([text, label, image])
            print(label_x)
            print(type(text_x))
            print(type(str(text_x)))
            print("image shape ", image.shape)
            plt.imshow(image)
            plt.show()
        coord.request_stop()
        coord.join(threads)



