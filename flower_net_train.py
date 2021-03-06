# -*-coding:utf-8-*-

from utils import flowers
from preprocess.load_batch_data import load_batch
from preprocess import convert_flowers_to_tfrecord
from utils import download_dataset
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import os
import tensorflow as tf

NUMBER_OF_STEPS = 10000
BATCH_SIZE = 64
cwd = os.path.dirname(os.path.abspath(__file__))

_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
FLOWERS_DATA_DIR = os.path.join(cwd, 'datasets/flower_photos')
TRAIN_DIR = os.path.join(cwd, 'model/train/flower_photos/')
CHECKPOINT_EXCLUDE_SCOPES = ["resnet_v2_50/logits"]
PRETRAIN_DIR = os.path.join(cwd, 'model/pretrain/')


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
        variables_to_restore)


def main(_):

    # download and conver flower_photos dataset to tfrecord
    download_dataset.maybe_download_and_extract(FLOWERS_DATA_DIR, _DATA_URL)
    convert_flowers_to_tfrecord.run(FLOWERS_DATA_DIR)
    
    with tf.Graph().as_default() as g:
        tf.logging.set_verbosity(tf.logging.INFO)
        dataset = flowers.get_split('train', FLOWERS_DATA_DIR)
        images, _, labels = load_batch(dataset, batch_size=BATCH_SIZE,
                                       is_training=True)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(images,
                                               num_classes=dataset.num_classes,
                                               is_training=True)
            logits = tf.squeeze(tf.convert_to_tensor(logits, tf.float32))

        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                               logits=logits)
        slim.losses.add_loss(loss)
        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('losses/total_loss', total_loss)

        # 梯度根性只更新最后一层
        var2training = slim.get_trainable_variables(scope="resnet_v2_50/logits")
        optimizer = tf.train.AdamOptimizer()
        train_op = slim.learning.create_train_op(total_loss, optimizer,
                                                 variables_to_train=var2training)

        init_fn = get_init_fn(checkpoint_exclude_scopes=CHECKPOINT_EXCLUDE_SCOPES,
                              checkpoint_dir=PRETRAIN_DIR)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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
