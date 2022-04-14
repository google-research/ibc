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

"""Conv spatial attention."""
import gin
import tensorflow.compat.v2 as tf


@gin.configurable
class SpatialAttentionEncoder(tf.keras.layers.Layer):
  """Resnet + spatial softmax encoder."""

  def __init__(self,
               filters,
               num_blocks=8,
               num_heads=64,
               target_height=90,
               target_width=120):
    """normalizer should be ['Batch', 'Layer', None]."""
    super(SpatialAttentionEncoder, self).__init__()

    self.target_height = target_height
    self.target_width = target_width
    self.num_heads = num_heads

    # Coordinates.
    self.coords = self._get_coords(target_height, target_width)

    # Define spatial softmax.
    self.spatial_softmax = SpatialSoftmax()

    # Encoder.
    self.conv0 = conv(filters, 3)
    self.blocks = [ResNetConvBlock(filters) for i in range(num_blocks)]
    self.conv1 = conv(num_heads, 1)

  def _get_coords(self, height, width):
    self.posx, self.posy = tf.meshgrid(
        tf.linspace(-1., 1., num=height),
        tf.linspace(-1., 1., num=width),
        indexing='ij')
    image_coords = tf.stack((self.posx, self.posy), axis=2)
    return image_coords[tf.newaxis, :, :, :]

  def _concat_coords(self, obs, batch_size):
    image_coords = tf.broadcast_to(self.coords,
                                   (batch_size,
                                    self.target_height,
                                    self.target_width, 2))
    obs = tf.concat((obs, image_coords), axis=-1)
    return obs

  def call(self, x):
    batch_size = tf.shape(x)[0]
    x = self._concat_coords(x, batch_size)
    x = self.conv0(x)
    for i in range(len(self.blocks)):
      x = self.blocks[i](x)
    x = self.conv1(x)

    # Dimensionality reduction.
    x = self.spatial_softmax(x)
    return x


def conv(filters, kernel_size):
  return tf.keras.layers.Conv2D(filters, kernel_size, padding='same',
                                activation=None)


class ResNetConvBlock(tf.keras.layers.Layer):
  """Convolutional resnet block."""

  def __init__(self, filters):
    super(ResNetConvBlock, self).__init__()
    self.conv0 = conv(filters // 4, 1)
    self.conv1 = conv(filters // 4, 3)
    self.conv2 = conv(filters, 1)
    self.conv3 = conv(filters, 1)

    self.activation0 = tf.keras.layers.ReLU()
    self.activation1 = tf.keras.layers.ReLU()
    self.activation2 = tf.keras.layers.ReLU()
    self.activation3 = tf.keras.layers.ReLU()

  def call(self, x):
    y = self.conv0(self.activation0(x))
    y = self.conv1(self.activation1(y))
    y = self.conv2(self.activation2(y))
    if x.shape != y.shape:
      x = self.conv3(self.activation3(x))
    return x + y


class SpatialSoftmax(tf.keras.layers.Layer):
  """Keras spatial soft-argmax layer. http://arxiv.org/abs/1509.06113."""

  def __init__(self):
    super(SpatialSoftmax, self).__init__()
    self.temperature = tf.Variable(initial_value=1., trainable=True,
                                   name='spatial_softmax_temperature')

  def call(self, features):
    """Computes the spatial soft-argmax over a convolutional feature map."""
    _, height, width, num_channels = features.shape
    # Create tensors for x and y coordinate values, scaled to range [-1, 1].
    pos_x, pos_y = tf.meshgrid(tf.linspace(-1., 1., num=height),
                               tf.linspace(-1., 1., num=width), indexing='ij')
    pos_x = tf.reshape(pos_x, [height * width])
    pos_y = tf.reshape(pos_y, [height * width])
    features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]),
                          [-1, height * width])
    softmax_attention = tf.nn.softmax(features / self.temperature)
    expected_x = tf.reduce_sum(pos_x * softmax_attention, [1], keepdims=True)
    expected_y = tf.reduce_sum(pos_y * softmax_attention, [1], keepdims=True)
    expected_xy = tf.concat([expected_x, expected_y], 1)
    feature_keypoints = tf.reshape(expected_xy, [-1, num_channels * 2])
    return feature_keypoints
