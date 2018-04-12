# -*-coding:utf-8-*-

import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from utils import flowers
from preprocess.load_batch_data import load_batch
from datetime import datetime
from tensorflow.contrib import slim
import time
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
FLOWERS_DATA_DIR = os.path.join(cwd, 'datasets/flower_photos')
TRAIN_DIR = os.path.join(cwd, 'model/train/flower_photos/')
TRAIN_OR_VAL = 'validation'
EVAL_DIR = os.path.join(cwd, 'model/eval/flower_photos/')
NUM_VALIDATION = 350
BATCH_SIZE = 64


def main(_):

    with tf.Graph().as_default() as g:

        tf.logging.set_verbosity(tf.logging.INFO)
        dataset = flowers.get_split(TRAIN_OR_VAL, FLOWERS_DATA_DIR)
        images, images_raw, labels = load_batch(dataset, batch_size=BATCH_SIZE,
                                                is_training=False)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(images,
                                               num_classes=dataset.num_classes,
                                               is_training=False)
            logits = tf.squeeze(tf.convert_to_tensor(logits, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            with slim.queues.QueueRunners(sess):
                writer = tf.summary.FileWriter(EVAL_DIR, g)
                step = 0
                while step < 100:
                    true_count = 0
                    num_iter = int(np.ceil(NUM_VALIDATION/BATCH_SIZE))
                    total_sample_count = num_iter * BATCH_SIZE
                    sess.run(tf.local_variables_initializer())
                    checkpoint_name = tf.train.latest_checkpoint(TRAIN_DIR)
                    init_fn = slim.assign_from_checkpoint_fn(os.path.join(TRAIN_DIR, checkpoint_name),
                                                             slim.get_model_variables())
                    init_fn(sess)

                    top_k_op = tf.nn.in_top_k(logits, labels, 1)
                    for _ in range(num_iter):
                        predictions = sess.run([top_k_op])
                        true_count += np.sum(predictions)
                        step += 1
                    precision = true_count / total_sample_count
                    tf.summary.scalar("precision", precision)
                    print('%s: accuracy = %.3f' % (datetime.now(), precision))

                    step += 1
                    time.sleep(10)
                writer.close()


if __name__ == "__main__":
    tf.app.run()
