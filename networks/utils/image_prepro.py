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

"""Shared utilities for image preprocessing."""
import tensorflow as tf


def stack_images_channelwise(obs, batch_size):
  # Use static shapes for hist, width, height, and channels since TPUs prefer
  # static shapes for some image ops. The batch size passed in may still be
  # dynamic.
  nhist = obs.get_shape()[1]
  nw = obs.get_shape()[2]
  nh = obs.get_shape()[3]
  nc = obs.get_shape()[4]
  obs = tf.reshape(obs, tf.concat([[batch_size], [nw, nh, nc * nhist]], axis=0))
  return obs


def preprocess(images, target_height, target_width):
  """Converts to [0,1], stacks, resizes."""
  # Scale to [0, 1].
  images = tf.image.convert_image_dtype(images, dtype=tf.float32)

  # Stack images channel-wise.
  batch_size = tf.shape(images)[0]
  images = stack_images_channelwise(images, batch_size)

  # Resize to target height and width.
  images = tf.image.resize(images, [target_height, target_width])
  return images
