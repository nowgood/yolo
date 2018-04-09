# -*-coding:utf-8-*-

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import os
import tensorflow as tf


class Resnetv2ToFlowerNet(object):

    def __init__(self, num_classes, is_training=True):
        self._num_classes = num_classes
        self._is_training = is_training

    def logits_fn(self, processed_images):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(processed_images, self._num_classes,
                                               is_training=self._is_training)
            logits = tf.convert_to_tensor(logits, tf.float32)
            return tf.squeeze(logits)

    def get_init_fn(self, checkpoint_dir, checkpoint_exclude_scopes=None):
        """Returns a function run by the chief worker to warm-start the training."""\

        if checkpoint_exclude_scopes is None:
            checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)
            return slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, checkpoint_name),
                                                  slim.get_model_variables())

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
            variables_to_restore)

