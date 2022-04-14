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

"""Utilities to precompute statistics on a dataset."""

import gin
import numpy as np
import tensorflow as tf

from tf_agents.networks import nest_map
from tf_agents.utils import tensor_normalizer

import tqdm


class ChanRunningStatistics:
  """Implements Chan's algorithm.

  For more details, see the parallel algorithm of Chan et al. at:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  """

  def __init__(self, initial_mean):
    self._n = 0
    if len(initial_mean.shape) > 1:
      self._mean = initial_mean[0, 0, :] * 0
    else:
      self._mean = initial_mean * 0
    self._m2 = 0
    self._m2_b_c = 0

  def update_running_statistics(self, sample):
    """Applies Chan's update rule to the running statistics."""
    sample_n = 1

    if len(sample.shape) > 1:
      # Compute statistics for last dim only. Reshape to turn other dims into
      # batch dim.
      sample = np.reshape(sample, [-1, sample.shape[-1]])
      sample_n = sample.shape[0]

      avg_sample, var_sample = tf.nn.moments(
          tf.convert_to_tensor(sample), axes=[0])
      avg_sample = avg_sample.numpy()
      var_sample = var_sample.numpy()
      m2_sample = var_sample * sample_n

    else:
      avg_sample = sample
      m2_sample = 0.0

    self._n, self._mean, self._m2, self._m2_b_c = (
        tensor_normalizer.parallel_variance_calculation(
            sample_n,
            avg_sample,
            m2_sample,
            self._n,
            self._mean,
            self._m2,
            self._m2_b_c,
        ))

  @property
  def mean(self):
    return self._mean

  @property
  def std(self):
    return np.sqrt(self.variance)

  @property
  def variance(self):
    return self._m2 / self._n

  @property
  def n(self):
    return self._n


def _action_update(action, min_action, max_action):
  """Updates the action statistics."""
  action = action.numpy()

  # Adding a batch dimension so that numpy can do per-dimension min
  action = action[None, Ellipsis]
  if min_action is None:
    min_action = action.min(axis=0)
    max_action = action.max(axis=0)
  else:
    min_action = np.minimum(min_action, action.min(axis=0))
    max_action = np.maximum(max_action, action.max(axis=0))

  return min_action, max_action


@gin.configurable
def compute_dataset_statistics(dataset, num_samples, nested_obs=True,
                               nested_actions=False,
                               min_max_actions=False,
                               use_sqrt_std=False):
  """Uses Chan's algorithm to compute mean, std in a single pass.

  Args:
    dataset: Dataset to compute statistics on. Should return batches of (obs,
      action) tuples. If `nested` is not True obs, and actions should be
      flattened first.
    num_samples: Number of samples to take from the dataset.
    nested_obs: If True generates a nest of norm layers matching the obs
      structures.
    nested_actions: If True generates a nest of norm layers matching the action
      structures.
    min_max_actions: If True, use [-1,1] instead of 0-mean, unit-var.
    use_sqrt_std: If True, divide by sqrt(std) instead of std for normalization.

  Returns:
    obs_norm_layer, per-dimension normalizer to 0-mean, unit-variance
    act_norm_layer, per-dimension normalizer to 0-mean, unit-variance
    min_action, shape [dim_A], per-dimension max actions in dataset
    max_action, shape [dim_A], per-dimension min actions in dataset
  """
  obs_statistics = None
  act_statistics = None

  def identity(x):
    return x

  def sqrt(x):
    return np.sqrt(x)

  if use_sqrt_std:
    get = sqrt
  else:
    get = identity

  with tqdm.tqdm(
      desc="Computing Dataset Statistics", total=num_samples) as progress_bar:

    observation = None
    action = None

    for observation, action in dataset.unbatch().take(num_samples):
      flat_obs = tf.nest.flatten(observation)
      flat_actions = tf.nest.flatten(action)

      if obs_statistics is None:
        # Initialize all params
        num_obs = len(flat_obs)
        num_act = len(flat_actions)

        if not nested_obs and num_obs > 1:
          raise ValueError("Found too many observations, make sure you set "
                           "`nested=True` or you flatten them.")

        if not nested_actions and num_act > 1:
          raise ValueError("Found too many actions, make sure you set "
                           "`nested=True` or you flatten them.")

        # [0] on the observation to take single value out of time dim.
        obs_statistics = [ChanRunningStatistics(o[0].numpy()) for o in flat_obs]
        act_statistics = [
            ChanRunningStatistics(a.numpy()) for a in flat_actions
        ]

        min_actions = [None for _ in range(num_act)]
        max_actions = [None for _ in range(num_act)]

      for obs, obs_stat in zip(flat_obs, obs_statistics):
        # Iterate over time dim.
        for o in obs:
          obs_stat.update_running_statistics(o.numpy())

      for act, act_stat in zip(flat_actions, act_statistics):
        act_stat.update_running_statistics(act.numpy())

      min_actions, max_actions = zip(*tf.nest.map_structure(
          _action_update,
          flat_actions,
          min_actions,
          max_actions,
          check_types=False))

      progress_bar.update(1)

  assert obs_statistics[0].n > 2

  obs_norm_layers = []
  act_norm_layers = []
  act_denorm_layers = []
  for obs_stat in obs_statistics:
    obs_norm_layers.append(
        StdNormalizationLayer(mean=obs_stat.mean, std=get(obs_stat.std)))

  for act_stat in act_statistics:
    if not min_max_actions:
      act_norm_layers.append(
          StdNormalizationLayer(mean=act_stat.mean, std=get(act_stat.std)))
      act_denorm_layers.append(
          StdDenormalizationLayer(mean=act_stat.mean, std=get(act_stat.std)))
    else:
      act_norm_layers.append(
          MinMaxNormalizationLayer(vmin=min_actions[0], vmax=max_actions[0]))
      act_denorm_layers.append(
          MinMaxDenormalizationLayer(vmin=min_actions[0], vmax=max_actions[0]))

  if nested_obs:
    obs_norm_layers = nest_map.NestMap(
        tf.nest.pack_sequence_as(observation, obs_norm_layers))
  else:
    obs_norm_layers = obs_norm_layers[0]

  if nested_actions:
    act_norm_layers = nest_map.NestMap(
        tf.nest.pack_sequence_as(action, act_norm_layers))
    act_denorm_layers = nest_map.NestMap(
        tf.nest.pack_sequence_as(action, act_denorm_layers))

    min_actions = tf.nest.pack_sequence_as(action, min_actions)
    max_actions = tf.nest.pack_sequence_as(action, max_actions)
  else:
    act_norm_layers = act_norm_layers[0]
    act_denorm_layers = act_denorm_layers[0]
    min_actions = min_actions[0]
    max_actions = max_actions[0]

  # Initialize act_denorm_layers:
  act_denorm_layers(min_actions)
  return (obs_norm_layers, act_norm_layers, act_denorm_layers, min_actions,
          max_actions)


EPS = np.finfo(np.float32).eps


class IdentityLayer(tf.keras.layers.Layer):

  def __init__(self, cast_dtype):
    super(IdentityLayer, self).__init__(trainable=False)
    self.cast_dtype = cast_dtype

  def __call__(self, x, **kwargs):
    return tf.cast(x, self.cast_dtype)


class StdNormalizationLayer(tf.keras.layers.Layer):
  """Maps an un-normalized vector to zmuv."""

  def __init__(self, mean, std):
    super(StdNormalizationLayer, self).__init__(trainable=False)
    self._mean = mean.astype(np.float32)
    self._std = std.astype(np.float32)

  def __call__(self, vector, **kwargs):
    vector = tf.cast(vector, tf.float32)
    return (vector - self._mean) / tf.maximum(self._std, EPS)


class StdDenormalizationLayer(tf.keras.layers.Layer):
  """Maps a zmuv-normalized vector back to its original mean and std."""

  def __init__(self, mean, std):
    super(StdDenormalizationLayer, self).__init__(trainable=False)
    self._mean = mean.astype(np.float32)
    self._std = std.astype(np.float32)

  def __call__(self, vector, mean_offset=True, **kwargs):
    vector = tf.cast(vector, tf.float32)
    result = (vector * tf.maximum(self._std, EPS))
    if mean_offset:
      result += self._mean
    return result


class MinMaxLayer(tf.keras.layers.Layer):

  def __init__(self, vmin, vmax):
    super(MinMaxLayer, self).__init__(trainable=False)
    self._min = vmin.astype(np.float32)
    self._max = vmax.astype(np.float32)
    self._mean_range = (self._min + self._max) / 2.0
    self._half_range = (0.5*(self._max - self._min))
    # Half_range shouldn't already be negative.
    self._half_range = tf.maximum(self._half_range, EPS)


class MinMaxNormalizationLayer(MinMaxLayer):
  """Maps an un-normalized vector to -1, 1."""

  def __call__(self, vector, **kwargs):
    vector = tf.cast(vector, tf.float32)
    return (vector - self._mean_range) / self._half_range


class MinMaxDenormalizationLayer(MinMaxLayer):
  """Maps -1, 1 vector back to un-normalized."""

  def __call__(self, vector, **kwargs):
    vector = tf.cast(vector, tf.float32)
    return (vector * self._half_range) + self._mean_range
