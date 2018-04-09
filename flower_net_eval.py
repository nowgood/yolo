# -*-coding:utf-8-*-

import os
import tensorflow as tf
from net.Resnetv2_50Variant import Resnetv2ToFlowerNet
from utils import flowers
from preprocess.get_minibatch_input import load_batch
from datetime import datetime
import time
from tensorflow.contrib import slim

cwd = os.path.dirname(os.path.abspath(__file__))
print("文件目录: ", cwd)

IMAGE_SIZE = 224
NUMBER_OF_STEPS = 10000
BATCH_SIZE = 128

_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
FLOWERS_DATA_DIR = os.path.join(cwd, 'datasets/flower_photos')
TRAIN_DIR = os.path.join(cwd, 'model/train/flower_photos/')
TRAIN_OR_VAL = 'validation'
EVAL_DIR =  os.path.join(cwd, 'model/eval/flower_photos/')


def main(_):

    with tf.Graph().as_default() as g:

        tf.logging.set_verbosity(tf.logging.INFO)
        dataset = flowers.get_split(TRAIN_OR_VAL, FLOWERS_DATA_DIR)
        images, images_raw, labels = load_batch(dataset, batch_size=BATCH_SIZE)
        net = Resnetv2ToFlowerNet(num_classes=dataset.num_classes, is_training=False)
        logits = net.logits_fn(images)
        prediction = tf.nn.softmax(logits)
        init_fn = net.get_init_fn(checkpoint_dir=TRAIN_DIR)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.initialize_local_variables())
                init_fn(sess)
                print(labels)
                pre, np_images_raw, np_labels = sess.run([prediction, images_raw, labels])
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre, 1), np_labels), tf.float32))
                print(accuracy)
                print('%s: accuracy @ 1 = %.3f' % (datetime.now(), accuracy))


if __name__ == "__main__":
    tf.app.run()
