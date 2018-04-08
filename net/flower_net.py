# -*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.contrib import slim

from net.Resnetv2_50Variant import Resnetv2ToFlowerNet
import os
import sys
from utils import flowers
from preprocess.get_minibatch_input import load_batch
from preprocess import download_and_convert_flowers

sys.path.append('./')
cwd = os.getcwd()
print(cwd)

IMAGE_SIZE = 224
CHECKPOINTS_DIR = "../model/pretrain/"
TRAIN_DIR = '../model/train/flower/'
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
FLOWERS_DATA_DIR = '../datasets/flower/'
CHECKPOINT_EXCLUDE_SCOPES = ["resnet_v2_50/logits"]


def main(_):

    # download and conver flower dataset to tfrecord
    download_and_convert_flowers.run(FLOWERS_DATA_DIR)
    
    with tf.Graph().as_default():

        dataset = flowers.get_split('train', FLOWERS_DATA_DIR)
        images, _, labels = load_batch(dataset, is_training=True)
        net = Resnetv2ToFlowerNet(CHECKPOINT_EXCLUDE_SCOPES, num_classes=dataset.num_classes)

        logits = net.logits_fn(images)
        print(logits.shape)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        print(one_hot_labels.shape)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('losses/Total Loss', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=TRAIN_DIR,
            init_fn=net.get_init_fn(),
            number_of_steps=1)

        print('Finished training. Last batch loss %f' % final_loss)


if __name__ == "__main__":
    tf.app.run()