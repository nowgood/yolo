# -*-coding:utf-8-*-

import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from utils import flowers
from preprocess.get_minibatch_input import load_batch
from datetime import datetime
from tensorflow.contrib import slim
import time

cwd = os.path.dirname(os.path.abspath(__file__))
print("文件目录: ", cwd)


FLOWERS_DATA_DIR = os.path.join(cwd, 'datasets/flower_photos')
TRAIN_DIR = os.path.join(cwd, 'model/train/flower_photos/')
TRAIN_OR_VAL = 'validation'
EVAL_DIR = os.path.join(cwd, 'model/eval/flower_photos/')

BATCH_SIZE = 128


def main(_):

    with tf.Graph().as_default() as g:

        tf.logging.set_verbosity(tf.logging.INFO)
        dataset = flowers.get_split(TRAIN_OR_VAL, FLOWERS_DATA_DIR)
        images, images_raw, labels = load_batch(dataset, batch_size=BATCH_SIZE)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(images,
                                               num_classes=dataset.num_classes,
                                               is_training=True)
            logits = tf.squeeze(tf.convert_to_tensor(logits, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.summary.FileWriter(EVAL_DIR, g)

        with tf.Session(config=config) as sess:
            with slim.queues.QueueRunners(sess):
                while True:
                    prediction = tf.nn.softmax(logits)
                    checkpoint_name = tf.train.latest_checkpoint(TRAIN_DIR)
                    init_fn = slim.assign_from_checkpoint_fn(os.path.join(TRAIN_DIR, checkpoint_name),
                                                             slim.get_model_variables())

                    sess.run(tf.initialize_local_variables())
                    init_fn(sess)
                    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                    acc = sess.run(accuracy)
                    tf.summary.scalar("accuracy", acc)
                    print('%s: accuracy @ 1 = %.3f' % (datetime.now(), acc))
                    time.sleep(60)


if __name__ == "__main__":
    tf.app.run()
