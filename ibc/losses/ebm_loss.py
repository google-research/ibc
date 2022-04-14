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

"""EBM loss functions."""

import tensorflow as tf


def info_nce(predictions,
             batch_size,
             num_counter_examples,
             softmax_temperature,
             kl):
  """EBM loss: can you classify the correct example?

  Args:
    predictions: [B x n+1] with true in column [:, -1]
    batch_size: B
    num_counter_examples: n
    softmax_temperature: the temperature of the softmax.
    kl: a KL Divergence loss object

  Returns:
    (loss per each element in the batch, and an optional
     dictionary with any loss objects to log)
  """
  softmaxed_predictions = tf.nn.softmax(
      predictions / softmax_temperature, axis=-1)

  # [B x n+1] with 1 in column [:, -1]
  indices = tf.ones(
      (batch_size,), dtype=tf.int32) * num_counter_examples
  labels = tf.one_hot(indices, depth=num_counter_examples + 1)

  per_example_loss = kl(labels, softmaxed_predictions)
  return per_example_loss, dict()


def cd(predictions):
  """World's simplest EBM loss function: contrastive divergence.

  Args:
    predictions: [B x n+1] with true in column [:, -1]
  Returns:
    (loss per each element in the batch, and an optional
       dictionary with any loss objects to log)
  """
  energy_data = predictions[:, -1:]
  energy_samp = predictions[:, :-1]
  per_example_loss = ((tf.reduce_mean(energy_samp, axis=1) -
                       tf.reduce_mean(energy_data, axis=1)))
  return per_example_loss, dict()


def cd_kl(predictions,
          counter_example_actions,
          predictions_copy
          ):
  """Improved CD loss, with KL, from: https://arxiv.org/pdf/2012.01316.pdf .

  Args:
    predictions: [B x n+1] with true in column [:, -1],
      with gradient flow through network but not mcmc
    counter_example_actions: [B x n x act_spec], with gradient flow through mcmc
    predictions_copy: [B x n+1] with true in column [:, -1], with gradient flow
      through mcmc but not network
  Returns:
    (loss per each element in the batch, and an optional
       dictionary with any loss objects to log)
  """
  energy_data = predictions[:, -1:]
  energy_samp = predictions[:, :-1]
  cd_per_example_loss = ((tf.reduce_mean(energy_samp, axis=1) -
                          tf.reduce_mean(energy_data, axis=1)))
  debug_dict = dict()
  debug_dict['cd_loss'] = tf.reduce_mean(cd_per_example_loss)

  energy_samp_copy = predictions_copy[:, :-1]
  dist = tf.reduce_sum(tf.square(
      (counter_example_actions[:, None, :, Ellipsis] -
       counter_example_actions[:, :, None, Ellipsis])), axis=-1)
  entropy_temperature = 1e-1
  entropy = -tf.math.exp(-entropy_temperature * dist)
  kl_per_example_loss = tf.reduce_mean(-energy_samp_copy[Ellipsis, None] - entropy,
                                       axis=[-2, -1])
  debug_dict['kl_loss'] = tf.reduce_mean(kl_per_example_loss)

  per_example_loss = cd_per_example_loss + kl_per_example_loss
  return per_example_loss, debug_dict


def clipped_cd(predictions,
               counter_example_actions,
               actions_size_n,
               soft):
  """An idea where 'close-enough' counter examples are treated softly.

  Args:
    predictions: [B x n+1] with true in column [:, -1]
    counter_example_actions: [B x n x act_spec]
    actions_size_n:  [B x n x act_spec]
    soft: bool, whether or not to apply soft local cone
  Returns:
    (loss per each element in the batch, and an optional
       dictionary with any loss objects to log)
  """
  debug_dict = dict()
  energy_data = predictions[:, -1:]  # [B x 1]
  energy_samp = predictions[:, :-1]  # [B x n]

  energy_data = tf.broadcast_to(energy_data, tf.shape(energy_samp))  # [B x n]

  thresh = 2e-1
  errors = tf.norm(counter_example_actions - actions_size_n,
                   axis=-1)**2  # [B x n]
  outside_thresh = tf.where(errors > thresh,
                            tf.ones_like(errors),
                            tf.zeros_like(errors))

  samp_minus_data = (energy_samp - energy_data) * outside_thresh
  per_example_loss = tf.reduce_mean(samp_minus_data, axis=1)

  if soft:
    w_shaping = 1.0
    inside_threshold = tf.where(errors <= thresh,
                                tf.ones_like(errors),
                                tf.zeros_like(errors))
    residual = (energy_data - energy_samp) - errors
    soft_loss = w_shaping * tf.reduce_mean(
        tf.abs(residual) * inside_threshold, axis=1)
    debug_dict['cd_loss'] = tf.reduce_mean(per_example_loss)
    debug_dict['soft_loss'] = tf.reduce_mean(soft_loss)
    per_example_loss += soft_loss

  debug_dict['fraction_outside_threshold'] = tf.reduce_mean(outside_thresh)
  return per_example_loss, debug_dict
