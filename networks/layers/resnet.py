# coding=utf-8
# Copyright 2022 The Reach ML Authors.
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

"""Keras layer for doing ResNet."""
import gin
import tensorflow.compat.v2 as tf


class Swish(tf.keras.layers.Layer):
  """A non-trainable, beta=1 swish layer."""

  def call(self, inputs):
    return tf.keras.activations.swish(inputs)


@gin.configurable
class ResNetLayer(tf.keras.layers.Layer):
  """ResNet layer, see variants at https://arxiv.org/pdf/1603.05027.pdf."""

  def __init__(
      self, hidden_sizes, rate, kernel_initializer, bias_initializer, dense,
      activation='relu', normalizer=None, make_weight_fn=None, **kwargs):
    """normalizer should be ['Batch', 'Layer', None]."""
    super(ResNetLayer, self).__init__(**kwargs)
    self.normalizer = normalizer

    # ResNet wants layers to be even numbers,
    # but remember there will be an additional
    # layer just to project to the first hidden size.
    assert len(hidden_sizes) % 2 == 0
    self._projection_layer = dense(
        hidden_sizes[0],
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

    self._weight_layers = []
    self._norm_layers = []
    self._activation_layers = []
    self._dropouts = []

    self._weight_layers_2 = []
    self._norm_layers_2 = []
    self._activation_layers_2 = []
    self._dropouts_2 = []

    def create_dense_layer(width):
      return dense(width,
                   activation=None,
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    if make_weight_fn is None:
      make_weight_fn = create_dense_layer

    # Step every other
    for l in range(0, len(hidden_sizes), 2):
      self._weight_layers.append(make_weight_fn(hidden_sizes[l]))
      if self.normalizer == 'Batch':
        self._norm_layers.append(tf.keras.layers.BatchNormalization())
      elif self.normalizer == 'Layer':
        self._norm_layers.append(
            tf.keras.layers.LayerNormalization(epsilon=1e-6))
      elif self.normalizer is None:
        pass
      else:
        raise ValueError('Expected a different normalizer.')
      if activation == 'relu':
        self._activation_layers.append(tf.keras.layers.ReLU())
      elif activation == 'swish':
        self._activation_layers.append(Swish())
      else:
        raise ValueError('Expected a different layer activation.')
      self._dropouts.append(tf.keras.layers.Dropout(rate))

      self._weight_layers_2.append(make_weight_fn(hidden_sizes[l+1]))
      if self.normalizer == 'Batch':
        self._norm_layers_2.append(tf.keras.layers.BatchNormalization())
      elif self.normalizer == 'Layer':
        self._norm_layers_2.append(
            tf.keras.layers.LayerNormalization(epsilon=1e-6))
      elif self.normalizer is None:
        pass
      else:
        raise ValueError('Expected a different normalizer.')
      if activation == 'relu':
        self._activation_layers_2.append(tf.keras.layers.ReLU())
      elif activation == 'swish':
        self._activation_layers_2.append(Swish())
      else:
        raise ValueError('Expected a different layer activation.')
      self._dropouts_2.append(tf.keras.layers.Dropout(rate))


class ResNetOrigLayer(ResNetLayer):
  """ResNet layer, original version."""

  def call(self, x, training):
    # Project to same size.
    x = self._projection_layer(x)

    # Do forward pass through resnet layers.
    for l in range(len(self._weight_layers)):
      x_start_block = tf.identity(x)
      x = self._weight_layers[l](x, training=training)
      if self.normalizer is not None:
        x = self._norm_layers[l](x, training=training)
      x = self._activation_layers[l](x, training=training)
      x = self._dropouts[l](x, training=training)

      x = self._weight_layers_2[l](x, training=training)
      if self.normalizer is not None:
        x = self._norm_layers_2[l](x, training=training)
      x = x_start_block + x
      x = self._activation_layers_2[l](x, training=training)
      x = self._dropouts_2[l](x, training=training)
    return x


class ResNetPreActivationLayer(ResNetLayer):
  """ResNet layer, improved 'pre-activation' version."""

  def call(self, x, training):
    x = self._projection_layer(x)

    # Do forward pass through resnet layers.
    for l in range(len(self._weight_layers)):
      x_start_block = tf.identity(x)
      if self.normalizer is not None:
        x = self._norm_layers[l](x, training=training)
      x = self._activation_layers[l](x, training=training)
      x = self._dropouts[l](x, training=training)
      x = self._weight_layers[l](x, training=training)

      if self.normalizer is not None:
        x = self._norm_layers_2[l](x, training=training)
      x = self._activation_layers_2[l](x, training=training)
      x = self._dropouts_2[l](x, training=training)
      x = self._weight_layers_2[l](x, training=training)
      x = x_start_block + x
    return x
