# -*-coding:utf-8-*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image2example(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(image),
        "image/label": int64_feature(label)
    }))


def image2tfrecord(image_paths, tfrecord_path):

    with tf.python_io.TFRecordWriter(tfrecord_path) as tf_writer:
        for idx, ele in enumerate(image_paths):
            with tf.gfile.Open(ele, "rb") as f:
                image_raw = f.read()
                example = image2example(image_raw, idx)
                tf_writer.write(example.SerializeToString())


def tfrecord2example():
    return {
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string),
        "image/label": tf.FixedLenFeature([], dtype=tf.int64)
     }


if __name__ == "__main__":
    image1_path = "../test_image/image1.jpg"
    image2_path = "../test_image/image2.jpg"
    tfrecord_path = "../test_image/image1.tfrecords"

    image2tfrecord([image1_path, image2_path], tfrecord_path)

    tf_reader = tf.TFRecordReader()

    """
    tf.train.string_input_producer() 输入为文件列表, 记得加 []
    该函数创建输入文件队列, 返回 A queue with the output strings.  
    A `QueueRunner` for the Queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

    然后可以通过 tf.train.batch() tf.train.shuffle_batch() 可以通过 num_threads
    指定多个线程同时执行入队操作, tf.train.shuffle_batch() 的入队操作就是数据读取和预处理的过程
    多个线程会同时读取一个文件的不同样例并进行预处理
    """

    file_queue = tf.train.string_input_producer([tfrecord_path])

    # key 就是文件的路径名称, 如 "b'../test_image/image1.tfrecords:0'"
    key, image_raw = tf_reader.read(file_queue)

    features_map = tfrecord2example()
    features = tf.parse_single_example(image_raw, features=features_map)

    label = features["image/label"]
    image = features["image/encoded"]

    print('after features["image/encoded"]:  ', image.shape)
    image = tf.image.decode_jpeg(image, channels=3)
    print("after tf.image.decode_jpeg:  ", image.shape)
    image = tf.image.resize_image_with_crop_or_pad(image, 400, 400)

    # 使用 tf.train.batch() image 必须有明确的 shape, 所以需要预处理设置 shape
    image, label = tf.train.batch([image, label], batch_size=2, num_threads=2)
    with tf.Session() as sess:
        # print(sess.run([key]))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(1):
            image, label = sess.run([image, label])
            print("label:  ", label)
            print("after sess.run:  ", image.shape)
            for ele in range(1):
                # 只用有一个位置为-1,
                # 不能写成 ele = np.reshape(image[ele, :, :, :], [-1, -1, 3])
                ele = np.reshape(image[ele, :, :, :], [400, -1, 3])
                print(ele.shape)
                plt.imshow(ele)
                # plt.axis("off")
                plt.show()

        coord.request_stop()
        coord.join(threads)




