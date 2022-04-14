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
import tensorflow_probability as tfp
from tf_agents.networks import network

tfd = tfp.distributions


@gin.configurable
class MLPMDN(network.Network):
  """MLP-MDN compatible with tfagents."""

  def __init__(self,
               obs_spec,
               action_spec,
               width=512,
               depth=2,
               rate=0.1,
               act_denorm_layer=None,
               num_components=1,
               training_temperature=2.5,
               test_temperature=2.5,
               test_variance_exponent=1.,
               name='MLPMDN',
               layers='MLPDropout'):
    super(MLPMDN, self).__init__(
        input_tensor_spec=obs_spec, state_spec=(), name=name)

    # For inference time, use to denormalize mdn action output.
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

    self.num_components = num_components

    self.action_size = action_spec.shape[0]
    self.mu = tf.keras.layers.Dense(
        (self.action_size * num_components),
        kernel_initializer='normal',
        bias_initializer='normal')
    self.logvar = tf.keras.layers.Dense(
        (self.action_size * num_components),
        kernel_initializer='normal',
        bias_initializer='normal')
    self.pi = tf.keras.layers.Dense(
        num_components,
        kernel_initializer='normal',
        bias_initializer='normal')
    self.training_temp = training_temperature
    self.test_temp = test_temperature
    self.test_variance_exponent = test_variance_exponent

  def call(self, obs, training, step_type=(), network_state=()):
    # Combine dict of observations to concatenated tensor. [B x T x obs_spec]
    obs = tf.concat(tf.nest.flatten(obs), axis=-1)

    # Flatten obs across time: [B x T * obs_spec]
    batch_size = tf.shape(obs)[0]
    x = tf.reshape(obs, [batch_size, -1])

    # Forward mlp.
    x = self._mlp(x, training=training)

    # Project to params.
    mu = self.mu(x)
    var = tf.exp(self.logvar(x))
    if not training:
      var = var**self.test_variance_exponent
    pi = self.pi(x)
    temp = self.training_temp if training else self.test_temp
    pi = pi / temp

    # Reshape into MDN distribution.
    batch_size = tf.shape(mu)[0]
    param_shape = [batch_size, self.num_components, self.action_size]
    mu = tf.reshape(mu, param_shape)
    var = tf.reshape(var, param_shape)

    if not training:
      mu = self._act_denorm_layer(mu)
      var = self._act_denorm_layer(var, mean_offset=False)

    components_distribution = tfd.MultivariateNormalDiag(loc=mu, scale_diag=var)
    x = tfd.MixtureSameFamily(
        tfd.Categorical(logits=pi), components_distribution)

    return x, network_state
