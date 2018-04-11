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


def tfrecord2image(tfrecord_path):

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
    features_map = {
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string),
        "image/label": tf.FixedLenFeature([], dtype=tf.int64)
    }

    features = tf.parse_single_example(image_raw, features=features_map)
    label = features["image/label"]
    image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
    print(image.shape)
    tf.expand_dims(image, 0)
    image = tf.image.resize_images(image, [400, 400], method=1)
    tf.squeeze(image)

    print(image.shape)
    return image, key, label


if __name__ == "__main__":
    image1_path = "../test_image/image1.jpg"
    image2_path = "../test_image/image2.jpg"
    tfrecord_path = "../test_image/image1.tfrecords"

    image2tfrecord([image1_path, image2_path], tfrecord_path)

    image, image_key, label = tfrecord2image(tfrecord_path)
    images, labels = tf.train.batch([image, label], batch_size=4, num_threads=2)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(1):
            image_x, image_key_x, label_x = sess.run([images, image_key, labels])
            print(label_x, " ", image_key_x)

            print(image_x.shape)
            for ele in range(4):
                ele = np.reshape(image_x[ele, :, :, :], [400, 400, 3])
                plt.imshow(ele)
                # plt.axis("off")
                plt.show()

        coord.request_stop()
        coord.join(threads)

        with tf.gfile.Open(image1_path, "rb") as f:
            image_raw = f.read()
            image_data = tf.image.decode_jpeg(image_raw, channels=3)
            print(image_data.shape)
            print("测试 a")
            # tf.image.resize_images() 各种方法显示看不懂,
            # 还是使用 tf.image.resize_image_with_crop_or_pad 比较好
            rs_image = tf.image.resize_image_with_crop_or_pad(image_data, 600, 600)
            rs_image = sess.run(rs_image)
            plt.imshow(rs_image)
            plt.show()




