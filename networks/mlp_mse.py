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

"""Implements a tf_agents compatible mlp-mse."""
import gin
from ibc.networks.layers import mlp_dropout
from ibc.networks.layers import resnet
import tensorflow as tf
from tf_agents.networks import network


@gin.configurable
class MLPMSE(network.Network):
  """MLP-MSE compatible with tfagents."""

  def __init__(self,
               obs_spec,
               action_spec,
               width=512,
               depth=2,
               rate=0.1,
               act_denorm_layer=None,
               name='MLPMSE',
               layers='MLPDropout'):
    super(MLPMSE, self).__init__(
        input_tensor_spec=obs_spec, state_spec=(), name=name)

    # For inference time, use to denormalize mse action output.
    self._act_denorm_layer = act_denorm_layer

    # Define MLP.
    hidden_sizes = [width for _ in range(depth)]
    dense = tf.keras.layers.Dense
    if layers == 'MLPDropout':
      self._mlp = mlp_dropout.MLPDropoutLayer(
          hidden_sizes, rate, kernel_initializer='normal',
          bias_initializer='normal', dense=dense)
    elif layers == 'ResNetOrig':
      self._mlp = resnet.ResNetOrigLayer(
          hidden_sizes, rate, kernel_initializer='normal',
          bias_initializer='normal', dense=dense)
    elif layers == 'ResNetPreActivation':
      self._mlp = resnet.ResNetPreActivationLayer(
          hidden_sizes, rate, kernel_initializer='normal',
          bias_initializer='normal', dense=dense)

    # Define projection to action.
    self._project_action = tf.keras.layers.Dense(
        action_spec.shape[-1],
        kernel_initializer='normal',
        bias_initializer='normal')

  def call(self, obs, training, step_type=(), network_state=()):
    # Combine dict of observations to concatenated tensor. [B x T x obs_spec]
    obs = tf.concat(tf.nest.flatten(obs), axis=-1)

    # Flatten obs across time: [B x T * obs_spec]
    batch_size = tf.shape(obs)[0]
    x = tf.reshape(obs, [batch_size, -1])

    # Forward mlp.
    x = self._mlp(x, training=training)

    # Project to action.
    x = self._project_action(x, training=training)

    if not training:
      x = self._act_denorm_layer(x)
    return x, network_state
