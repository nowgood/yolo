# -*-coding:utf-8-*-
from six.moves import urllib
import os
import sys
import tarfile
import tensorflow as tf


def maybe_download_and_extract(dataset_dir, dataset_url):
    """Download and extract the tarball datasets"""

    dest_directory = dataset_dir
    if not os.path.exists(dest_directory):
       os.makedirs(dest_directory)

    filename = dataset_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(dataset_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def clean_up_temporary_files(dataset_dir, dataset_url):
    """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
    filename = dataset_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)
    tmp_dir = os.path.join(dataset_dir, filename)
    tf.gfile.DeleteRecursively(tmp_dir)
