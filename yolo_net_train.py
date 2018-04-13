# -*-coding:utf-8-*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from net import yolo_net
from preprocess import yolonet_preprocessing
from loss.losses import batch_loss
from dataset_tools.tfrecord_ops_for_objectdetection import tfrecord2example
from tensorflow.contrib import slim


cwd = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = 224
NUMBER_OF_STEPS = 10000
TRAIN_DIR = os.path.join(cwd, 'model/train/voc2007/')
CHECKPOINT_EXCLUDE_SCOPES = ["resnet_v2_50/logits"]
PRETRAIN_DIR = os.path.join(cwd, 'model/pretrain/')
OUTPUT_DIR = os.path.join(cwd, "datasets/voc2007/pascal_voc.tfrecords")

flags = tf.app.flags
flags.DEFINE_string("tfrecord_path", OUTPUT_DIR, "path to tfrecord file")
flags.DEFINE_string("batch_size", "64", "batch size")
flags.DEFINE_string("logdir", TRAIN_DIR, "batch size")
FLAGS = flags.FLAGS


def get_init_fn(checkpoint_dir, checkpoint_exclude_scopes=None):
    """Returns a function run by the chief worker to warm-start the training."""

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoint_dir, 'resnet_v2_50.ckpt'),
        variables_to_restore, ignore_missing_vars=True)


def main(_):

    tf_reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([FLAGS.tfrecord_path])
    _, image_raw = tf_reader.read(file_queue)
    features_map = tfrecord2example()
    features = tf.parse_single_example(image_raw, features=features_map)

    label = features['image/object/class/label']
    image = features['image/encoded']
    xmin = features['image/object/bbox/xmin']
    ymin = features['image/object/bbox/ymin']
    xmax = features['image/object/bbox/xmax']
    ymax = features['image/object/bbox/ymax']
    height = features['image/height']
    width = features['image/width']
    height = tf.to_int32(height)
    width = tf.to_int32(width)
    label = tf.to_int32(label)
    label = label.values
    bbox = tf.concat([xmin.values, ymin.values, xmax.values, ymax.values], axis=0)
    bbox = tf.reshape(bbox, [-1, 4])
    label = tf.reshape(label, [-1])

    image = tf.image.decode_jpeg(image, channels=3)
    image = yolonet_preprocessing.preprocess(image, output_height=IMAGE_SIZE,
                                             output_width=IMAGE_SIZE)
    bbox = yolonet_preprocessing.scale(bbox, tf.truediv(IMAGE_SIZE, height), tf.truediv(IMAGE_SIZE, width))

    bbox = tf.expand_dims(bbox, axis=0)
    label = tf.expand_dims(label, axis=0)
    image = tf.expand_dims(image, axis=0)

    '''
    image, bbox, label = tf.train.shuffle_batch([image, bbox, label],
                                                batch_size=FLAGS.batch_size,
                                                capacity=int(FLAGS.batch_size) * 3 + 500,
                                                num_threads=8,
                                                min_after_dequeue=500)
    '''
    prediction = yolo_net.yolonet(image, is_training=True)
    yolo_loss = batch_loss(prediction, bbox, label, batch_size=1)
    print(yolo_loss.shape)

    tf.summary.scalar('losses/yolo_loss', yolo_loss)
    res_loss = tf.losses.get_total_loss()
    total_loss = res_loss + yolo_loss

    tf.summary.scalar('losses/total_loss', total_loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = slim.learning.create_train_op(total_loss, optimizer, check_numerics=False)
    init_fn = get_init_fn(checkpoint_exclude_scopes=CHECKPOINT_EXCLUDE_SCOPES,
                          checkpoint_dir=PRETRAIN_DIR)

    tf.summary.merge_all()

    print("start training")
    final_loss = slim.learning.train(
        train_op,
        logdir=TRAIN_DIR,
        init_fn=init_fn,
        number_of_steps=NUMBER_OF_STEPS,
        trace_every_n_steps=500,
        log_every_n_steps=50,
        session_config=config,
        save_interval_secs=60)

    print('Finished training. Last batch loss %f' % final_loss)


if __name__ == "__main__":
    tf.app.run()
