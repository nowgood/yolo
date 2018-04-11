# -*-coding:utf-8-*-

import tensorflow as tf
import matplotlib.pyplot as plt


def image2tfrecord(image_path, tfrecord_path):

    with tf.gfile.Open(image_path, "rb") as f:
        image_raw = f.read()

    label = 20
    height = 2.0

    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def floatlist_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    example = tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(image_raw),
        "image/label": int64_feature(label),
        "image/height": floatlist_feature(height)
    }))

    tf_writer = tf.python_io.TFRecordWriter(tfrecord_path)
    tf_writer.write(example.SerializeToString())
    tf_writer.close()


def tfrecord2image(tfrecord_path):

    tf_reader = tf.TFRecordReader()
    # tf.train.string_input_producer() 输入为文件列表, 记得加 []
    file_queue = tf.train.string_input_producer([tfrecord_path])
    image_key, image_raw = tf_reader.read(file_queue)
    features = {
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string),
        "image/label": tf.FixedLenFeature([], dtype=tf.int64)
    }

    features = tf.parse_single_example(image_raw, features=features)
    image = tf.image.decode_image(features["image/encoded"])
    label = features["image/label"]
    return image, image_key, label


if __name__ == "__main__":
    image_path = "../test_image/image1.jpg"
    tfrecord_path = "../test_image/image1.tfrecords"
    image2tfrecord(image_path, tfrecord_path)
    image, image_key, label = tfrecord2image(tfrecord_path)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image, image_key, label = sess.run([image, image_key, label])

        print(image_key, " ", label)

        plt.imshow(image)
        plt.axis("off")
        plt.show()
        coord.request_stop()
        coord.join(threads)

