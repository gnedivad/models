# # Copyright 2017 Google Inc.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Task towers for PixelDA model."""
# import tensorflow as tf

# slim = tf.contrib.slim


# def add_task_specific_model(images,
#                             hparams,
#                             num_classes=10,
#                             is_training=False,
#                             reuse_private=False,
#                             private_scope=None,
#                             reuse_shared=False,
#                             shared_scope=None):
#   """Create a classifier for the given images.
#   The classifier is composed of a few 'private' layers followed by a few
#   'shared' layers. This lets us account for different image 'style', while
#   sharing the last few layers as 'content' layers.
#   Args:
#     images: A `Tensor` of size [batch_size, height, width, 3].
#     hparams: model hparams
#     num_classes: The number of output classes.
#     is_training: whether model is training
#     reuse_private: Whether or not to reuse the private weights, which are the
#       first few layers in the classifier
#     private_scope: The name of the variable_scope for the private (unshared)
#       components of the classifier.
#     reuse_shared: Whether or not to reuse the shared weights, which are the last
#       few layers in the classifier
#     shared_scope: The name of the variable_scope for the shared components of
#       the classifier.
#   Returns:
#     The logits, a `Tensor` of shape [batch_size, num_classes].
#   Raises:
#     ValueError: If hparams.task_classifier is an unknown value
#   """

#   model = hparams.task_tower
#   # Make sure the classifier name shows up in graph
#   shared_scope = shared_scope or (model + '_shared')
#   kwargs = {
#       'num_classes': num_classes,
#       'is_training': is_training,
#       'reuse_private': reuse_private,
#       'reuse_shared': reuse_shared,
#   }

#   if private_scope:
#     kwargs['private_scope'] = private_scope
#   if shared_scope:
#     kwargs['shared_scope'] = shared_scope

#   quaternion_pred = None
#   with slim.arg_scope(
#       [slim.conv2d, slim.fully_connected],
#       activation_fn=tf.nn.relu,
#       weights_regularizer=tf.contrib.layers.l2_regularizer(
#           hparams.weight_decay_task_classifier)):
#     with slim.arg_scope([slim.conv2d], padding='SAME'):
#       if model == 'doubling_pose_estimator':
#         logits, quaternion_pred = doubling_cnn_class_and_quaternion(
#             images, num_private_layers=hparams.num_private_layers, **kwargs)
#       elif model == 'mnist':
#         logits, _ = mnist_classifier(images, **kwargs)
#       elif model == 'svhn':
#         logits, _ = svhn_classifier(images, **kwargs)
#       elif model == 'gtsrb':
#         logits, _ = gtsrb_classifier(images, **kwargs)
#       elif model == 'pose_mini':
#         logits, quaternion_pred = pose_mini_tower(images, **kwargs)
#       else:
#         raise ValueError('Unknown task classifier %s' % model)

#   return logits, quaternion_pred


# #####################################
# # Classifiers used in the DSN paper #
# #####################################


# def mnist_classifier(images,
#                      is_training=False,
#                      num_classes=10,
#                      reuse_private=False,
#                      private_scope='mnist',
#                      reuse_shared=False,
#                      shared_scope='task_model'):
#   """Creates the convolutional MNIST model from the gradient reversal paper.
#   Note that since the output is a set of 'logits', the values fall in the
#   interval of (-infinity, infinity). Consequently, to convert the outputs to a
#   probability distribution over the characters, one will need to convert them
#   using the softmax function:
#         logits, endpoints = conv_mnist(images, is_training=False)
#         predictions = tf.nn.softmax(logits)
#   Args:
#     images: the MNIST digits, a tensor of size [batch_size, 28, 28, 1].
#     is_training: specifies whether or not we're currently training the model.
#       This variable will determine the behaviour of the dropout layer.
#     num_classes: the number of output classes to use.
#   Returns:
#     the output logits, a tensor of size [batch_size, num_classes].
#     a dictionary with key/values the layer names and tensors.
#   """

#   net = {}

#   with tf.variable_scope(private_scope, reuse=reuse_private):
#     net['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
#     net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')

#   with tf.variable_scope(shared_scope, reuse=reuse_shared):
#     net['conv2'] = slim.conv2d(net['pool1'], 48, [5, 5], scope='conv2')
#     net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
#     net['fc3'] = slim.fully_connected(
#         slim.flatten(net['pool2']), 100, scope='fc3')
#     net['fc4'] = slim.fully_connected(
#         slim.flatten(net['fc3']), 100, scope='fc4')
#     logits = slim.fully_connected(
#         net['fc4'], num_classes, activation_fn=None, scope='fc5')
#   return logits, net


# def svhn_classifier(images,
#                     is_training=False,
#                     num_classes=10,
#                     reuse_private=False,
#                     private_scope=None,
#                     reuse_shared=False,
#                     shared_scope='task_model'):
#   """Creates the convolutional SVHN model from the gradient reversal paper.
#   Note that since the output is a set of 'logits', the values fall in the
#   interval of (-infinity, infinity). Consequently, to convert the outputs to a
#   probability distribution over the characters, one will need to convert them
#   using the softmax function:
#         logits = mnist.Mnist(images, is_training=False)
#         predictions = tf.nn.softmax(logits)
#   Args:
#     images: the SVHN digits, a tensor of size [batch_size, 40, 40, 3].
#     is_training: specifies whether or not we're currently training the model.
#       This variable will determine the behaviour of the dropout layer.
#     num_classes: the number of output classes to use.
#   Returns:
#     the output logits, a tensor of size [batch_size, num_classes].
#     a dictionary with key/values the layer names and tensors.
#   """

#   net = {}

#   with tf.variable_scope(private_scope, reuse=reuse_private):
#     net['conv1'] = slim.conv2d(images, 64, [5, 5], scope='conv1')
#     net['pool1'] = slim.max_pool2d(net['conv1'], [3, 3], 2, scope='pool1')

#   with tf.variable_scope(shared_scope, reuse=reuse_shared):
#     net['conv2'] = slim.conv2d(net['pool1'], 64, [5, 5], scope='conv2')
#     net['pool2'] = slim.max_pool2d(net['conv2'], [3, 3], 2, scope='pool2')
#     net['conv3'] = slim.conv2d(net['pool2'], 128, [5, 5], scope='conv3')

#     net['fc3'] = slim.fully_connected(
#         slim.flatten(net['conv3']), 3072, scope='fc3')
#     net['fc4'] = slim.fully_connected(
#         slim.flatten(net['fc3']), 2048, scope='fc4')

#     logits = slim.fully_connected(
#         net['fc4'], num_classes, activation_fn=None, scope='fc5')

#   return logits, net


# def gtsrb_classifier(images,
#                      is_training=False,
#                      num_classes=43,
#                      reuse_private=False,
#                      private_scope='gtsrb',
#                      reuse_shared=False,
#                      shared_scope='task_model'):
#   """Creates the convolutional GTSRB model from the gradient reversal paper.
#   Note that since the output is a set of 'logits', the values fall in the
#   interval of (-infinity, infinity). Consequently, to convert the outputs to a
#   probability distribution over the characters, one will need to convert them
#   using the softmax function:
#         logits = mnist.Mnist(images, is_training=False)
#         predictions = tf.nn.softmax(logits)
#   Args:
#     images: the SVHN digits, a tensor of size [batch_size, 40, 40, 3].
#     is_training: specifies whether or not we're currently training the model.
#       This variable will determine the behaviour of the dropout layer.
#     num_classes: the number of output classes to use.
#     reuse_private: Whether or not to reuse the private components of the model.
#     private_scope: The name of the private scope.
#     reuse_shared: Whether or not to reuse the shared components of the model.
#     shared_scope: The name of the shared scope.
#   Returns:
#     the output logits, a tensor of size [batch_size, num_classes].
#     a dictionary with key/values the layer names and tensors.
#   """

#   net = {}

#   with tf.variable_scope(private_scope, reuse=reuse_private):
#     net['conv1'] = slim.conv2d(images, 96, [5, 5], scope='conv1')
#     net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')
#   with tf.variable_scope(shared_scope, reuse=reuse_shared):
#     net['conv2'] = slim.conv2d(net['pool1'], 144, [3, 3], scope='conv2')
#     net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
#     net['conv3'] = slim.conv2d(net['pool2'], 256, [5, 5], scope='conv3')
#     net['pool3'] = slim.max_pool2d(net['conv3'], [2, 2], 2, scope='pool3')

#     net['fc3'] = slim.fully_connected(
#         slim.flatten(net['pool3']), 512, scope='fc3')
#     logits = slim.fully_connected(
#         net['fc3'], num_classes, activation_fn=None, scope='fc4')

#     return logits, net


# #########################
# # pose_mini task towers #
# #########################


# def pose_mini_tower(images,
#                     num_classes=11,
#                     is_training=False,
#                     reuse_private=False,
#                     private_scope='pose_mini',
#                     reuse_shared=False,
#                     shared_scope='task_model'):
#   """Task tower for the pose_mini dataset."""

#   with tf.variable_scope(private_scope, reuse=reuse_private):
#     net = slim.conv2d(images, 32, [5, 5], scope='conv1')
#     net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
#   with tf.variable_scope(shared_scope, reuse=reuse_shared):
#     net = slim.conv2d(net, 64, [5, 5], scope='conv2')
#     net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
#     net = slim.flatten(net)

#     net = slim.fully_connected(net, 128, scope='fc3')
#     net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
#     with tf.variable_scope('quaternion_prediction'):
#       quaternion_pred = slim.fully_connected(
#           net, 4, activation_fn=tf.tanh, scope='fc_q')
#       quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

#     logits = slim.fully_connected(
#         net, num_classes, activation_fn=None, scope='fc4')

#     return logits, quaternion_pred


# def doubling_cnn_class_and_quaternion(images,
#                                       num_private_layers=1,
#                                       num_classes=10,
#                                       is_training=False,
#                                       reuse_private=False,
#                                       private_scope='doubling_cnn',
#                                       reuse_shared=False,
#                                       shared_scope='task_model'):
#   """Alternate conv, pool while doubling filter count."""
#   net = images
#   depth = 32
#   layer_id = 1

#   with tf.variable_scope(private_scope, reuse=reuse_private):
#     while num_private_layers > 0 and net.shape.as_list()[1] > 5:
#       net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
#       net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
#       depth *= 2
#       layer_id += 1
#       num_private_layers -= 1

#   with tf.variable_scope(shared_scope, reuse=reuse_shared):
#     while net.shape.as_list()[1] > 5:
#       net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
#       net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
#       depth *= 2
#       layer_id += 1

#     net = slim.flatten(net)
#     net = slim.fully_connected(net, 100, scope='fc1')
#     net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
#     quaternion_pred = slim.fully_connected(
#         net, 4, activation_fn=tf.tanh, scope='fc_q')
#     quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

#     logits = slim.fully_connected(
#         net, num_classes, activation_fn=None, scope='fc_logits')

#     return logits, quaternion_pred


"""Task towers for PixelDA model."""
import tensorflow as tf

slim = tf.contrib.slim


def add_task_specific_model(real_images,
                            virtual_images,
                            hparams,
                            # num_classes=10,
                            is_training=False,
                            reuse_private=False,
                            private_scope=None,
                            reuse_shared=False,
                            shared_scope=None):
  """Create a classifier for the given images.

  The classifier is composed of a few 'private' layers followed by a few
  'shared' layers. This lets us account for different image 'style', while
  sharing the last few layers as 'content' layers.

  Args:
    images: A `Tensor` of size [batch_size, height, width, 3].
    hparams: model hparams
    # num_classes: The number of output classes.
    is_training: whether model is training
    reuse_private: Whether or not to reuse the private weights, which are the
      first few layers in the classifier
    private_scope: The name of the variable_scope for the private (unshared)
      components of the classifier.
    reuse_shared: Whether or not to reuse the shared weights, which are the last
      few layers in the classifier
    shared_scope: The name of the variable_scope for the shared components of
      the classifier.

  Returns:
    The logits, a `Tensor` of shape [batch_size, 6]. #  [batch_size, num_classes].

  Raises:
    ValueError: If hparams.task_classifier is an unknown value
  """

  model = hparams.task_tower
  # Make sure the classifier name shows up in graph
  shared_scope = shared_scope or (model + '_shared')
  kwargs = {
      # 'num_classes': num_classes,
      'is_training': is_training,
      'reuse_private': reuse_private,
      'reuse_shared': reuse_shared,
  }

  if private_scope:
    kwargs['private_scope'] = private_scope
  if shared_scope:
    kwargs['shared_scope'] = shared_scope

  quaternion_pred = None
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_regularizer=tf.contrib.layers.l2_regularizer(
          hparams.weight_decay_task_classifier)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      # if model == 'doubling_pose_estimator':
      #   logits, quaternion_pred = doubling_cnn_class_and_quaternion(
      #       images, num_private_layers=hparams.num_private_layers, **kwargs)
      # elif model == 'mnist':
      #   logits, _ = mnist_classifier(images, **kwargs)
      # elif model == 'svhn':
      #   logits, _ = svhn_classifier(images, **kwargs)
      # elif model == 'gtsrb':
      #   logits, _ = gtsrb_classifier(images, **kwargs)
      # elif model == 'pose_mini':
      #   logits, quaternion_pred = pose_mini_tower(images, **kwargs)
      if model == 'lung':
        logits = lung_classifier(real_images, virtual_images, **kwargs)
      else:
        raise ValueError('Unknown task classifier %s' % model)

  return logits  # , quaternion_pred


#####################################
# Classifiers used in the DSN paper #
#####################################


def mnist_classifier(images,
                     is_training=False,
                     num_classes=10,
                     reuse_private=False,
                     private_scope='mnist',
                     reuse_shared=False,
                     shared_scope='task_model'):
  """Creates the convolutional MNIST model from the gradient reversal paper.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits, endpoints = conv_mnist(images, is_training=False)
        predictions = tf.nn.softmax(logits)

  Args:
    images: the MNIST digits, a tensor of size [batch_size, 28, 28, 1].
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    num_classes: the number of output classes to use.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  net = {}

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net['conv2'] = slim.conv2d(net['pool1'], 48, [5, 5], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
    net['fc3'] = slim.fully_connected(
        slim.flatten(net['pool2']), 100, scope='fc3')
    net['fc4'] = slim.fully_connected(
        slim.flatten(net['fc3']), 100, scope='fc4')
    logits = slim.fully_connected(
        net['fc4'], num_classes, activation_fn=None, scope='fc5')
  return logits, net


def svhn_classifier(images,
                    is_training=False,
                    num_classes=10,
                    reuse_private=False,
                    private_scope=None,
                    reuse_shared=False,
                    shared_scope='task_model'):
  """Creates the convolutional SVHN model from the gradient reversal paper.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits = mnist.Mnist(images, is_training=False)
        predictions = tf.nn.softmax(logits)

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 40, 40, 3].
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    num_classes: the number of output classes to use.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  net = {}

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net['conv1'] = slim.conv2d(images, 64, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [3, 3], 2, scope='pool1')

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net['conv2'] = slim.conv2d(net['pool1'], 64, [5, 5], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [3, 3], 2, scope='pool2')
    net['conv3'] = slim.conv2d(net['pool2'], 128, [5, 5], scope='conv3')

    net['fc3'] = slim.fully_connected(
        slim.flatten(net['conv3']), 3072, scope='fc3')
    net['fc4'] = slim.fully_connected(
        slim.flatten(net['fc3']), 2048, scope='fc4')

    logits = slim.fully_connected(
        net['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, net


def gtsrb_classifier(images,
                     is_training=False,
                     num_classes=43,
                     reuse_private=False,
                     private_scope='gtsrb',
                     reuse_shared=False,
                     shared_scope='task_model'):
  """Creates the convolutional GTSRB model from the gradient reversal paper.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits = mnist.Mnist(images, is_training=False)
        predictions = tf.nn.softmax(logits)

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 40, 40, 3].
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    num_classes: the number of output classes to use.
    reuse_private: Whether or not to reuse the private components of the model.
    private_scope: The name of the private scope.
    reuse_shared: Whether or not to reuse the shared components of the model.
    shared_scope: The name of the shared scope.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  net = {}

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net['conv1'] = slim.conv2d(images, 96, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')
  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net['conv2'] = slim.conv2d(net['pool1'], 144, [3, 3], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
    net['conv3'] = slim.conv2d(net['pool2'], 256, [5, 5], scope='conv3')
    net['pool3'] = slim.max_pool2d(net['conv3'], [2, 2], 2, scope='pool3')

    net['fc3'] = slim.fully_connected(
        slim.flatten(net['pool3']), 512, scope='fc3')
    logits = slim.fully_connected(
        net['fc3'], num_classes, activation_fn=None, scope='fc4')

    return logits, net


#########################
# pose_mini task towers #
#########################


def pose_mini_tower(images,
                    num_classes=11,
                    is_training=False,
                    reuse_private=False,
                    private_scope='pose_mini',
                    reuse_shared=False,
                    shared_scope='task_model'):
  """Task tower for the pose_mini dataset."""

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
    net = slim.flatten(net)

    net = slim.fully_connected(net, 128, scope='fc3')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
    with tf.variable_scope('quaternion_prediction'):
      quaternion_pred = slim.fully_connected(
          net, 4, activation_fn=tf.tanh, scope='fc_q')
      quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc4')

    return logits, quaternion_pred


def doubling_cnn_class_and_quaternion(images,
                                      num_private_layers=1,
                                      num_classes=10,
                                      is_training=False,
                                      reuse_private=False,
                                      private_scope='doubling_cnn',
                                      reuse_shared=False,
                                      shared_scope='task_model'):
  """Alternate conv, pool while doubling filter count."""
  net = images
  depth = 32
  layer_id = 1

  with tf.variable_scope(private_scope, reuse=reuse_private):
    while num_private_layers > 0 and net.shape.as_list()[1] > 5:
      net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1
      num_private_layers -= 1

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    while net.shape.as_list()[1] > 5:
      net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1

    net = slim.flatten(net)
    net = slim.fully_connected(net, 100, scope='fc1')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
    quaternion_pred = slim.fully_connected(
        net, 4, activation_fn=tf.tanh, scope='fc_q')
    quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc_logits')

    return logits, quaternion_pred


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())


def building_block(inputs, filters, is_training, projection_shortcut, strides):
  """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides):
  """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1)

  return tf.identity(inputs, "output")


EMBEDDING_DEPTH = 64
FILTERS = [
  EMBEDDING_DEPTH / 8,
  EMBEDDING_DEPTH / 8,
  EMBEDDING_DEPTH / 4,
  EMBEDDING_DEPTH / 2,
  EMBEDDING_DEPTH,
]
LAYERS = [3, 4, 6, 3]


def residual_modules(inputs, in_filters, is_training):
  tf.identity(inputs, "residual_input")

  with tf.variable_scope("init_layer"):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=FILTERS[0], kernel_size=7, strides=1)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=1, padding='SAME')
    inputs = tf.identity(inputs, 'initial_max_pool')

  with tf.variable_scope("block_layer1"):
    inputs = block_layer(
        inputs=inputs, filters=FILTERS[1], block_fn=building_block,
        blocks=LAYERS[0], strides=1, is_training=is_training)

  with tf.variable_scope("block_layer2"):
    inputs = block_layer(
        inputs=inputs, filters=FILTERS[2], block_fn=building_block,
        blocks=LAYERS[1], strides=2, is_training=is_training)

  with tf.variable_scope("block_layer3"):
    inputs = block_layer(
        inputs=inputs, filters=FILTERS[3], block_fn=building_block,
        blocks=LAYERS[2], strides=2, is_training=is_training)

  with tf.variable_scope("block_layer4"):
    inputs = block_layer(
        inputs=inputs, filters=FILTERS[4], block_fn=building_block,
        blocks=LAYERS[3], strides=2, is_training=is_training)

  with tf.variable_scope("final_layer"):
    inputs = batch_norm_relu(inputs, is_training)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=4, strides=1, padding='VALID')
    inputs = tf.identity(inputs, 'final_avg_pool')
    # inputs = tf.reshape(inputs, [inputs.get_shape()[0].value, -1])
    inputs = tf.squeeze(inputs, axis=[1, 2])  # [?, 1, 1, 128] => [?, 128]

  return inputs


def lung_classifier(real_images,
                    virtual_images,
                    num_private_layers=1,
                    is_training=False,
                    reuse_private=False,
                    private_scope='lung',
                    reuse_shared=False,
                    shared_scope='task_model'):
  init_num_private_layers = num_private_layers

  net = real_images
  depth = 32
  layer_id = 1

  with tf.variable_scope(private_scope, reuse=reuse_private):
    while num_private_layers > 0 and net.shape.as_list()[1] > 5:
      net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1
      num_private_layers -= 1

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    while net.shape.as_list()[1] > 5:
      net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1

    net = slim.flatten(net)
    net = slim.fully_connected(net, 100, scope='fc1')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
    # quaternion_pred = slim.fully_connected(
    #     net, 4, activation_fn=tf.tanh, scope='fc_q')
    # quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

  num_private_layers = init_num_private_layers
  net2 = virtual_images
  depth = 32
  layer_id = 1
  with tf.variable_scope(private_scope, reuse=True):
    while num_private_layers > 0 and net2.shape.as_list()[1] > 5:
      net2 = slim.conv2d(net2, depth, [3, 3], scope='conv%s' % layer_id)
      net2 = slim.max_pool2d(net2, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1
      num_private_layers -= 1

  with tf.variable_scope(shared_scope, reuse=True):
    while net2.shape.as_list()[1] > 5:
      net2 = slim.conv2d(net2, depth, [3, 3], scope='conv%s' % layer_id)
      net2 = slim.max_pool2d(net2, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1

    net2 = slim.flatten(net2)
    net2 = slim.fully_connected(net2, 100, scope='fc1')
    net2 = slim.dropout(net2, 0.5, is_training=is_training, scope='dropout')
    # quaternion_pred = slim.fully_connected(
    #     net2, 4, activation_fn=tf.tanh, scope='fc_q')
    # quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

  if reuse_private or reuse_shared:
    tf.get_variable_scope().reuse_variables()

  logits = slim.fully_connected(
    tf.concat([net, net2], axis=1), 6, activation_fn=None, scope='fc_logits')

  return logits
  # x0 = real_images
  # x1 = virtual_images
  # in_filters = x0.shape[3]
  # with tf.variable_scope("cnn", reuse=reuse_shared):
  #   with tf.variable_scope("residual"):
  #     x0 = residual_modules(x0, in_filters, is_training)
  #     x0 = tf.identity(x0, name="x0")
  #     # Use this instead of `tf.variable_scope("residual", reuse=True)`
  #     # since creating a new scope with name variables "cnn/residual_1/cnn"
  #     tf.get_variable_scope().reuse_variables()
  #     x1 = residual_modules(x1, in_filters, is_training)
  #     x1 = tf.identity(x1, name="x1")

  #     # Concatenates x0 and x1
  #     x = tf.concat([x0, x1], axis=1)
  #     x = tf.identity(x, name="embeddings")
  #   with tf.variable_scope("fcl"):
  #     logits = tf.layers.dense(inputs=x, units=6)
  #     logits = tf.identity(logits, name="logits")  # cnn/fcl/logits
  #     # add_variable_summaries(logits)

  # return logits
