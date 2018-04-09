# -*-coding:utf-8-*-

import tensorflow as tf
import urllib.request as urllib
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from utils import imagenet
from preprocess import  inception_preprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils import download_dataset

sys.path.append('./')

image_size = 224
checkpoints_dir = "../model/pretrain/"

_CKPT_URL = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"

download_dataset.maybe_download_and_extract(checkpoints_dir, _CKPT_URL)


def main(_):
    with tf.Graph().as_default():
        url = 'https://upload.wikimedia.org/wikipedia/commons/5/5c/Tigershark3.jpg'
        image_string = urllib.urlopen(url).read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'resnet_v2_50.ckpt'),
            slim.get_model_variables('resnet_v2_50'))

        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = np.reshape(probabilities, [1001])
            print(probabilities.shape)
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
            print(sorted_inds)
        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.axis('off')
        plt.show()

        names = imagenet.create_readable_names_for_imagenet_labels()

        for i in range(5):
            index = sorted_inds[i]
            # Shift the index of a class name by one.
            print('Probability %0.6f%% => [%s]' % (probabilities[index] * 100, names[index + 1]))


if __name__ == "__main__":
    tf.app.run()