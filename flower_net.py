# -*-coding:utf-8-*-

import os
import tensorflow as tf
from tensorflow.contrib import slim
from net.Resnetv2_50Variant import Resnetv2ToFlowerNet
from utils import flowers
from preprocess.get_minibatch_input import load_batch
from preprocess import convert_flowers_to_tfrecord
from utils import download_dataset

cwd = os.path.dirname(os.path.abspath(__file__))
print("文件目录: ", cwd)

IMAGE_SIZE = 224
NUMBER_OF_STEPS = 10000
BATCH_SIZE = 128

_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
FLOWERS_DATA_DIR = os.path.join(cwd, 'datasets/flower_photos')
TRAIN_DIR = os.path.join(cwd, 'model/train/flower_photos/')
CHECKPOINTS_DIR = os.path.join(cwd, "model/pretrain/")
CHECKPOINT_EXCLUDE_SCOPES = ["resnet_v2_50/logits"]


def main(_):

    # download and conver flower_photos dataset to tfrecord
    download_dataset.maybe_download_and_extract(FLOWERS_DATA_DIR, _DATA_URL)
    convert_flowers_to_tfrecord.run(FLOWERS_DATA_DIR)
    
    with tf.Graph().as_default():

        dataset = flowers.get_split('train', FLOWERS_DATA_DIR)
        images, _, labels = load_batch(dataset, batch_size=BATCH_SIZE, is_training=True)
        net = Resnetv2ToFlowerNet(CHECKPOINT_EXCLUDE_SCOPES, num_classes=dataset.num_classes,
                                  checkpoint_dir=TRAIN_DIR)
        logits = net.logits_fn(images)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('losses/Total Loss', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=TRAIN_DIR,
            init_fn=net.get_init_fn(False),
            number_of_steps=NUMBER_OF_STEPS,
            trace_every_n_steps=50,
            log_every_n_steps=50)

        print('Finished training. Last batch loss %f' % final_loss)

if __name__ == "__main__":
    tf.app.run()