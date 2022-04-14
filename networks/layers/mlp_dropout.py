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

"""Keras layer for doing MLP + dropout."""
import tensorflow.compat.v2 as tf


class MLPDropoutLayer(tf.keras.layers.Layer):
  """MLP with dropout."""

  def __init__(
      self, hidden_sizes, rate, kernel_initializer, bias_initializer, dense,
      activation='relu', **kwargs):
    super(MLPDropoutLayer, self).__init__(**kwargs)
    self._fc_layers = []
    self._dropouts = []
    for l in range(len(hidden_sizes)):
      self._fc_layers.append(dense(
          hidden_sizes[l],
          activation=activation,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer))
      self._dropouts.append(tf.keras.layers.Dropout(rate))

  def call(self, x, training):
    # Do forward pass through mlp layers.
    for l in range(len(self._fc_layers)):
      x = self._fc_layers[l](x, training=training)
      x = self._dropouts[l](x, training=training)
    return x
