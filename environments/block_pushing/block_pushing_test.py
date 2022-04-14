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

"""Tests for ibc.environments.block_pushing."""

import collections
from ibc.environments.block_pushing import block_pushing
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import utils


class Blocks2DTest(tf.test.TestCase):

  def test_load_push_env(self):
    block_pushing.BlockPush()

  def test_load_push_env_insert(self):
    block_pushing.BlockPush(task=block_pushing.BlockTaskVariant.INSERT)

  def test_validate_environment(self):
    env = suite_gym.load('BlockPush-v0')
    utils.validate_py_environment(env)

  def test_push_normalized_env(self):
    env = block_pushing.BlockPushNormalized()
    random_action = np.random.random(size=env.action_space.shape) * 2 - 1
    random_action_convert = env.calc_normalized_action(
        env.calc_unnormalized_action(random_action))
    self.assertLess(np.abs(random_action_convert - random_action).max(), 1e-6)

    env.reset()
    state, _, _, _ = env.step(random_action)
    self.assertEqual(tuple(state.keys()),
                     tuple(env.observation_space.spaces.keys()))
    state_unnormalized = env.calc_unnormalized_state(state)
    self.assertEqual(tuple(state_unnormalized.keys()),
                     tuple(env._env.observation_space.spaces.keys()))
    state_convert = env.calc_normalized_state(state_unnormalized)
    self.assertEqual(tuple(state_convert.keys()), tuple(state.keys()))
    for key in state_convert.keys():
      self.assertLess(np.abs(state[key] - state_convert[key]).max(), 1e-6,
                      'mismatch on key %s (%s vs %s) ' % (
                          key, state[key], state_convert[key]))

  def test_rgb_environment(self):
    env = suite_gym.load('BlockPushRgb-v0')
    time_step = env.reset()
    self.assertEqual(time_step.observation['rgb'].shape, (240, 320, 3))

  def _assertEqualState(self,
                        state_a,
                        state_b):
    self.assertEqual(state_a.keys(), state_b.keys())  # Also checks ordering.
    for key, value in state_a.items():
      self.assertArrayNear(value, state_b[key], 1e-6,
                           f'state mismatch on key {key}')

  def _test_serialize_state(self, task):
    np.random.seed(0)
    env = block_pushing.BlockPush(task=task, seed=0)
    s0 = env.reset()
    pybullet_state = env.get_pybullet_state()
    obj_poses = [env.pybullet_client.getBasePositionAndOrientation(i) for i in
                 range(env.pybullet_client.getNumBodies())]

    # Now step the environment with a random action saving the next state.
    action = 0.1 * np.random.uniform(
        env.action_space.low, env.action_space.high)
    s1, reward, done, _ = env.step(action)

    # Now take another step followed by a reset to put the environment into
    # a new state.
    env.step(np.random.uniform(env.action_space.low, env.action_space.high))
    env.reset()

    # Now reset back to the serialized state and make sure states match.
    env.set_pybullet_state(pybullet_state)
    obj_poses_reset = [
        env.pybullet_client.getBasePositionAndOrientation(i) for i in
        range(env.pybullet_client.getNumBodies())]
    for (pos, quat), (pos_reset, quat_reset) in zip(obj_poses, obj_poses_reset):
      self.assertArrayNear(pos, pos_reset, 1e-6)
      self.assertArrayNear(quat, quat_reset, 1e-6)
    s0_reset = env.compute_state()
    self._assertEqualState(s0, s0_reset)

    # Also step the same action and make sure we end up at the same state.
    s1_reset, reward_reset, done_reset, _ = env.step(action)
    self._assertEqualState(s1, s1_reset)
    self.assertAlmostEqual(reward, reward_reset, places=6)
    self.assertEqual(done, done_reset)

  def test_serialize_state_push(self):
    self._test_serialize_state(block_pushing.BlockTaskVariant.PUSH)

  def test_serialize_state_reach(self):
    self._test_serialize_state(block_pushing.BlockTaskVariant.REACH)

  def test_serialize_state_insert(self):
    self._test_serialize_state(block_pushing.BlockTaskVariant.INSERT)


if __name__ == '__main__':
  tf.test.main()
