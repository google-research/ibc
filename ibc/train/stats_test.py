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

"""Tests for ibc.ibc.stats."""

from ibc.environments.block_pushing import block_pushing
from ibc.ibc.train import stats
import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.specs import array_spec

NUM_SAMPLES = 500
TIME_DIM = 2


def flatten_observation(obs, action):
  flat_obs = tf.nest.flatten(obs)
  flat_obs = tf.concat(flat_obs, axis=-1)
  return flat_obs, action


class StatsTest(tf.test.TestCase):

  def generate_dataset(self, images):
    image_size = None
    if images:
      image_size = (64, 64)

    env = block_pushing.BlockPush(
        task=block_pushing.BlockTaskVariant.PUSH, image_size=image_size)
    env = suite_gym.wrap_env(env)
    self._observation_spec = env.observation_spec()
    self._action_spec = env.action_spec()

    rng = np.random.RandomState(42)
    sample_obs = array_spec.sample_spec_nest(
        self._observation_spec, rng, outer_dims=(NUM_SAMPLES, TIME_DIM))
    sample_act = array_spec.sample_spec_nest(
        self._action_spec, rng, outer_dims=(NUM_SAMPLES,))
    data = (sample_obs, sample_act)
    dataset = tf.data.Dataset.from_tensors(
        tf.nest.map_structure(tf.convert_to_tensor, data))
    return dataset

  def test_check_num_obs(self):
    dataset = self.generate_dataset(images=False)

    with self.assertRaisesRegex(ValueError, 'too many'):
      stats.compute_dataset_statistics(dataset, num_samples=3, nested_obs=False)

  def test_norm_denorm(self):
    dataset = self.generate_dataset(images=False)
    dataset = dataset.map(flatten_observation)

    (_, act_norm_layers, act_denorm_layer, _, _) = (
        stats.compute_dataset_statistics(
            dataset, num_samples=NUM_SAMPLES, nested_obs=False))

    data = list(iter(dataset))[0]
    _, actions = data
    # Ensure normalized, then denormalized action == action.
    np.testing.assert_array_almost_equal(
        act_denorm_layer(act_norm_layers(actions)), actions, decimal=3)

  def test_flat_stats(self):
    dataset = self.generate_dataset(images=False)
    dataset = dataset.map(flatten_observation)

    (obs_norm_layers, act_norm_layers, _, min_actions, max_actions) = (
        stats.compute_dataset_statistics(
            dataset, num_samples=NUM_SAMPLES, nested_obs=False))

    data = list(iter(dataset))[0]
    observation, actions = data

    # Flatten time dim.
    observation = np.reshape(observation.numpy(), [NUM_SAMPLES * TIME_DIM, -1])
    obs_mean = np.mean(observation, axis=0)
    obs_std = np.std(observation, axis=0)

    np.testing.assert_almost_equal(
        obs_mean, obs_norm_layers._mean, decimal=3)
    np.testing.assert_almost_equal(
        obs_std,
        obs_norm_layers._std, decimal=3)

    act_mean = np.mean(actions.numpy(), axis=0)
    act_std = np.std(actions.numpy(), axis=0)
    act_min = np.min(actions.numpy(), axis=0)
    act_max = np.max(actions.numpy(), axis=0)

    np.testing.assert_almost_equal(
        act_mean, act_norm_layers._mean, decimal=3)
    np.testing.assert_almost_equal(
        act_std,
        act_norm_layers._std, decimal=3)

    np.testing.assert_almost_equal(act_min, min_actions)
    np.testing.assert_almost_equal(act_max, max_actions)

  def test_nested_obs_stats(self):
    dataset = self.generate_dataset(images=False)

    (obs_norm_layers, act_norm_layers, _, min_actions, max_actions) = (
        stats.compute_dataset_statistics(
            dataset, num_samples=NUM_SAMPLES, nested_obs=True))

    data = list(iter(dataset))[0]
    observations, actions = data

    # Flatten time dim.
    for observation, obs_norm_layer in zip(
        tf.nest.flatten(observations),
        tf.nest.flatten(obs_norm_layers._nested_layers)):
      observation = np.reshape(observation.numpy(),
                               [NUM_SAMPLES * TIME_DIM, -1])
      obs_mean = np.mean(observation, axis=0)
      obs_std = np.std(observation, axis=0)

      np.testing.assert_almost_equal(
          obs_mean,
          obs_norm_layer.layers[0]._mean,
          decimal=3)
      np.testing.assert_almost_equal(
          obs_std,
          obs_norm_layer.layers[0]._std,
          decimal=3)

    act_mean = np.mean(actions.numpy(), axis=0)
    act_std = np.std(actions.numpy(), axis=0)
    act_min = np.min(actions.numpy(), axis=0)
    act_max = np.max(actions.numpy(), axis=0)

    np.testing.assert_almost_equal(
        act_mean,
        act_norm_layers._mean,
        decimal=3)
    np.testing.assert_almost_equal(
        act_std,
        act_norm_layers._std,
        decimal=3)
    np.testing.assert_almost_equal(act_min, min_actions)
    np.testing.assert_almost_equal(act_max, max_actions)

  def test_nested_img_stats(self):
    dataset = self.generate_dataset(images=True)

    def float_and_scale_image(t):
      if t.dtype == tf.uint8:
        return tf.cast(t, tf.float32) / 255.0
      return t

    dataset = dataset.map(
        lambda o, a: tf.nest.map_structure(float_and_scale_image, (o, a)))

    (obs_norm_layers, act_norm_layers, _, min_actions, max_actions) = (
        stats.compute_dataset_statistics(
            dataset, num_samples=NUM_SAMPLES, nested_obs=True,
            nested_actions=True))

    data = list(iter(dataset))[0]
    observations, actions = data

    # Flatten time dim.
    for observation, obs_norm_layer in zip(
        tf.nest.flatten(observations),
        tf.nest.flatten(obs_norm_layers._nested_layers)):
      observation = np.reshape(observation.numpy(),
                               [-1, observation.shape[-1]])
      obs_mean = np.mean(observation, axis=0)
      obs_std = np.std(observation, axis=0)

      np.testing.assert_almost_equal(
          obs_mean,
          obs_norm_layer.layers[0]._mean,
          decimal=3)
      np.testing.assert_almost_equal(
          obs_std,
          obs_norm_layer.layers[0]._std,
          decimal=3)

    act_mean = np.mean(actions.numpy(), axis=0)
    act_std = np.std(actions.numpy(), axis=0)
    act_min = np.min(actions.numpy(), axis=0)
    act_max = np.max(actions.numpy(), axis=0)

    np.testing.assert_almost_equal(
        act_mean,
        act_norm_layers._nested_layers.layers[0]._mean,
        decimal=3)
    np.testing.assert_almost_equal(
        act_std,
        act_norm_layers._nested_layers.layers[0]._std,
        decimal=3)
    np.testing.assert_almost_equal(act_min, min_actions)
    np.testing.assert_almost_equal(act_max, max_actions)


if __name__ == '__main__':
  tf.test.main()
