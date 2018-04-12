# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
from lxml import etree
from six.moves import urllib
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'


def float_feature(values):
    """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


"""Utility functions for creating TFRecord data sets."""


def read_examples_list(path):
    """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def make_initializable_iterator(dataset):
    """Creates an iterator, and initializes tables.

  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return iterator


def read_dataset(file_read_func, decode_func, input_files, config):
    """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf.data.Dataset.interleave, to read
      every individual file into a tf.data.Dataset.
    decode_func: Function to apply to all records.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.

  Returns:
    A tf.data.Dataset based on config.
  """
    # Shard, shuffle, and read files.
    filenames = tf.concat([tf.matching_files(pattern) for pattern in input_files],
                          0)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if config.shuffle:
        filename_dataset = filename_dataset.shuffle(
            config.filenames_shuffle_buffer_size)
    filename_dataset = filename_dataset.repeat(config.num_epochs or None)

    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_func, cycle_length=config.num_readers, sloppy=True))
    if config.shuffle:
        records_dataset.shuffle(config.shuffle_buffer_size)
    tensor_dataset = records_dataset.map(
        decode_func, num_parallel_calls=config.num_parallel_map_calls)
    return tensor_dataset.prefetch(config.prefetch_size)


if __name__ == "__main__":

    # test recursive_parse_xml_to_dict()
    with tf.gfile.GFile("../data/annotation_test.xml", 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)['annotation']
    import json
    print(json.dumps(data))

    """output
    {
      "folder": "VOC2007",
      "filename": "007750.jpg",
      "source": {
        "database": "The VOC2007 Database",
        "annotation": "PASCAL VOC2007",
        "image": "flickr",
        "flickrid": "338109130"
      },
      "owner": {
        "flickrid": "gabrielrobichaud",
        "name": "Gabriel Robichaud"
      },
      "size": {
        "width": "500",
        "height": "375",
        "depth": "3"
      },
      "segmented": "0",
      "object": [
        {
          "name": "dog",
          "pose": "Unspecified",
          "truncated": "1",
          "difficult": "0",
          "bndbox": {
            "xmin": "128",
            "ymin": "23",
            "xmax": "500",
            "ymax": "375"
          }
        }
      ]
    }
    """

