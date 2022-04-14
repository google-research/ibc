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

"""Pixel-EBM supporting late fusion."""

import gin
from ibc.networks.layers import conv_maxpool
from ibc.networks.layers import dense_resnet_value
from ibc.networks.layers import spatial_attention_encoder
from ibc.networks.utils import image_prepro
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.utils import nest_utils


def get_encoder_network(encoder_network, target_height, target_width, channels):
  if encoder_network == 'SpatialAttentionEncoder':
    return spatial_attention_encoder.SpatialAttentionEncoder(
        target_height=target_height,
        target_width=target_width)
  elif encoder_network == 'ConvMaxpoolEncoder':
    return conv_maxpool.get_conv_maxpool(target_height, target_width, channels)
  else:
    raise ValueError('Unsupported encoder_network %s' % encoder_network)


def get_value_network(value_network):
  if value_network == 'DenseResnetValue':
    return dense_resnet_value.DenseResnetValue()
  else:
    raise ValueError('Unsupported value_network %s' % value_network)


@gin.configurable
class PixelEBM(network.Network):
  """Late fusion PixelEBM."""

  def __init__(self,
               obs_spec,
               action_spec,
               encoder_network,
               value_network,
               target_height=90,
               target_width=120,
               name='PixelEBM'):
    super(PixelEBM, self).__init__(
        input_tensor_spec=(obs_spec, action_spec),
        state_spec=(),
        name=name,
    )
    sequence_length = obs_spec['rgb'].shape[0]
    # We stack all images and coord-conv.
    num_channels = (3 * sequence_length)
    self._encoder = get_encoder_network(encoder_network,
                                        target_height,
                                        target_width,
                                        num_channels)
    self.target_height = target_height
    self.target_width = target_width
    self._value = get_value_network(value_network)

    rgb_shape = obs_spec['rgb'].shape
    self._static_height = rgb_shape[1]
    self._static_width = rgb_shape[2]
    self._static_channels = rgb_shape[3]

  def encode(self, obs, training):
    """Embeds images."""
    images = obs['rgb']

    # Ensure shape with static shapes from spec since shape information may
    # be lost in the data pipeline. ResizeBilinear is not supported with
    # dynamic shapes on TPU.
    # First 2 dims are batch size, seq len.
    images = tf.ensure_shape(images, [
        None, None, self._static_height, self._static_width,
        self._static_channels
    ])

    images = image_prepro.preprocess(images,
                                     target_height=self.target_height,
                                     target_width=self.target_width)
    observation_encoding = self._encoder(images, training=training)
    return observation_encoding

  def call(
      self,
      inputs,
      training,
      step_type=(),
      network_state=((), (), ()),
      observation_encoding=None):
    obs, act = inputs

    # If we pass in observation_encoding, we are doing late fusion.
    if observation_encoding is None:
      # Otherwise embed for the first time.
      observation_encoding = self.encode(obs, training)
      batch_size = tf.shape(obs['rgb'])[0]
      num_samples = tf.shape(act)[0] // batch_size
      observation_encoding = nest_utils.tile_batch(
          observation_encoding, num_samples)

    # Concat [obs, act].
    x = tf.concat([observation_encoding, act], -1)

    # Forward value network.
    x = self._value(x, training=training)

    # Squeeze extra dim.
    x = tf.squeeze(x, axis=-1)

    return x, network_state
