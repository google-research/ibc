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

# pylint: disable=invalid-name
# Keeping code style from original author (was given this class by imordatch@
# in a colab notebook.)
"""Keras layer for spectral norm.

Reference: https://arxiv.org/abs/1802.05957
Spectral normalization ensures Lipschitz continuity of the model.
"""
import tensorflow.compat.v2 as tf
K = tf.keras.backend


class DenseSN(tf.keras.layers.Dense):
  """Spectral norm dense layers."""

  def build(self, input_shape):
    assert len(input_shape) >= 2
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(shape=(input_dim, self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.u = self.add_weight(
        shape=tuple([1, self.kernel.shape.as_list()[-1]]),
        initializer=tf.keras.initializers.RandomNormal(0, 1),
        name='sn',
        trainable=False)
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=2, axes={-1: input_dim})
    self.built = True

  def call(self, inputs, training=None):
    """Forward the net."""
    def _l2normalize(v, eps=1e-12):
      return v / (K.sum(v ** 2) ** 0.5 + eps)
    def power_iteration(W, u):
      _u = u
      _v = _l2normalize(K.dot(_u, K.transpose(W)))
      _u = _l2normalize(K.dot(_v, W))
      return _u, _v
    W_shape = self.kernel.shape.as_list()
    # Flatten the Tensor
    W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
    _u, _v = power_iteration(W_reshaped, self.u)
    # Calculate Sigma
    sigma = K.dot(_v, W_reshaped)
    sigma = K.dot(sigma, K.transpose(_u))
    # normalize it
    W_bar = W_reshaped / sigma
    # reshape weight tensor
    if not training or training is None:
      W_bar = K.reshape(W_bar, W_shape)
    else:
      with tf.control_dependencies([self.u.assign(_u)]):
        W_bar = K.reshape(W_bar, W_shape)
    output = K.dot(inputs, W_bar)
    if self.use_bias:
      output = K.bias_add(output, self.bias, data_format='channels_last')
    if self.activation is not None:
      output = self.activation(output)
    return output
