from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Dependency imports
import tensorflow as tf

from slim.datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'lung_32_%s.tfrecord'

_SPLITS_TO_SIZES = {'train': 10000, 'valid': 100, 'test': 1000, 'uint8': 900, 'float32': 900}

_ITEMS_TO_DESCRIPTIONS = {
  'real_image': 'A [32 x 32 x 1] RGB image.',
  'virtual_image': 'A [32 x 32 x 1] RGB image.',
  'label': 'A float32 vector of length 6',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
    'real_image': tf.FixedLenFeature([], dtype=tf.string),
    'images/format': tf.FixedLenFeature([], dtype=tf.string, default_value='raw'),
    'virtual_image': tf.FixedLenFeature([], dtype=tf.string),
    'label': tf.FixedLenFeature([], dtype=tf.string),
  }

  items_to_handlers = {
    'real_image': slim.tfexample_decoder.Image(
      image_key='real_image',
      format_key='images/format',
      shape=[32, 32, 1],  # required after my hack in
                            # tensorflow/contrib/slim/python/slim/data/tfexample_decoder.py
      channels=1,           # unused after above hack
      dtype=tf.float32),
    'virtual_image': slim.tfexample_decoder.Image(
      image_key='virtual_image',
      format_key='images/format',
      shape=[32, 32, 1],
      channels=1,
      dtype=tf.float32),
    'label': slim.tfexample_decoder.Image(
      image_key='label',
      format_key='images/format',
      shape=[6],   # required
      channels=1,  # unused
      dtype=tf.float32),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      # num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)
