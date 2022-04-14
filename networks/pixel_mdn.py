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

"""Implements a tf_agents compatible mlp-mdn with conv+action layers underneath."""
import gin
from ibc.networks.layers import conv_maxpool
from ibc.networks.layers import mlp_dropout
from ibc.networks.layers import resnet
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network

tfd = tfp.distributions


@gin.configurable
class ConvMLPMDN(network.Network):
  """MLP-MDN compatible with tfagents."""

  def __init__(self,
               obs_spec,
               action_spec,
               width=512,
               depth=2,
               rate=0.1,
               act_denorm_layer=None,
               coord_conv=True,
               target_height=90,
               target_width=120,
               num_components=1,
               training_temperature=2.5,
               test_temperature=2.5,
               test_variance_exponent=1.,
               name='ConvMLPMDN',
               layers='MLPDropout'):
    super(ConvMLPMDN, self).__init__(
        input_tensor_spec=obs_spec, state_spec=(), name=name)
    dense = tf.keras.layers.Dense

    # Define Convnet.
    self.target_height = target_height
    self.target_width = target_width
    sequence_length = obs_spec['rgb'].shape[0]
    # We stack all images and coord-conv.
    num_channels = (3 * sequence_length)
    if coord_conv:
      self._init_coord_conv()
      num_channels += 2
    self._use_coord_conv = coord_conv

    self.cnn = conv_maxpool.get_conv_maxpool(
        self.target_height, self.target_width, num_channels)

    # Optionally use to denormalize mse action output.
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
    self.mu = tf.keras.layers.Dense((self.action_size * num_components),
                                    kernel_initializer='normal',
                                    bias_initializer='normal')
    self.logvar = tf.keras.layers.Dense((self.action_size * num_components),
                                        kernel_initializer='normal',
                                        bias_initializer='normal')
    self.pi = tf.keras.layers.Dense(
        num_components, kernel_initializer='normal', bias_initializer='normal')
    self.training_temp = training_temperature
    self.test_temp = test_temperature
    self.test_variance_exponent = test_variance_exponent

  def _init_coord_conv(self):
    posy, posx = tf.meshgrid(
        tf.linspace(-1., 1., num=self.target_height),
        tf.linspace(-1., 1., num=self.target_width),
        indexing='ij')
    self.image_coords = tf.stack((posy, posx), axis=2)  # (H, W, 2)

  def _stack_images_channelwise(self, obs, batch_size):
    nhist = tf.shape(obs)[1]
    nw = tf.shape(obs)[2]
    nh = tf.shape(obs)[3]
    nc = tf.shape(obs)[4]
    obs = tf.reshape(obs, [batch_size, nw, nh, nc * nhist])
    return obs

  def _concat_coordconv(self, obs, batch_size):
    image_coords = tf.broadcast_to(self.image_coords,
                                   (batch_size,
                                    self.target_height,
                                    self.target_width, 2))
    obs = tf.concat((obs, image_coords), axis=-1)
    return obs

  def call(self, inputs, training, step_type=(), network_state=()):
    obs = inputs['rgb']

    # Stack images channel-wise.
    batch_size = tf.shape(obs)[0]
    obs = self._stack_images_channelwise(obs, batch_size)

    # Resize to target height and width.
    obs = tf.image.resize(obs, [self.target_height, self.target_width])

    # Concat image with coord conv.
    if self._use_coord_conv:
      obs = self._concat_coordconv(obs, batch_size)

    # Forward cnn.
    x = self.cnn(obs)

    # Start of mdn.

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
