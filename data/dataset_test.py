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

"""Tests for ibc.data.dataset."""

import collections
import os
import tempfile
from typing import List

from ibc.data.dataset import filter_episodes
from ibc.data.dataset import load_tfrecord_dataset_sequence
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing.oracles import oriented_push_oracle as oriented_push_oracle_module
import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import example_encoding_dataset


class FilterEpisodesTest(tf.test.TestCase):

  def _build_traj(self, step_types):
    return Trajectory(step_type=tf.constant(step_types),
                      action=tf.range(len(step_types)),
                      observation=(),
                      policy_info=(),
                      next_step_type=(),
                      reward=(),
                      discount=())

  def test_no_first(self):
    sample = self._build_traj([StepType.MID] * 6)
    sample = filter_episodes(sample)
    self.assertEqual(sample.action.numpy().tolist(), list(range(6)))

  def test_first_at_start(self):
    sample = self._build_traj([StepType.FIRST] + 5 * [StepType.MID])
    sample = filter_episodes(sample)
    self.assertEqual(sample.action.numpy().tolist(), list(range(6)))

  def test_first_at_mid(self):
    sample = self._build_traj([StepType.FIRST, StepType.MID, StepType.LAST,
                               StepType.FIRST, StepType.MID, StepType.MID])
    sample = filter_episodes(sample)
    self.assertEqual(sample.action.numpy().tolist(), [3, 3, 3, 3, 4, 5])

  def test_first_at_end(self):
    sample = self._build_traj([StepType.FIRST] * 6)
    sample = filter_episodes(sample)
    self.assertEqual(sample.action.numpy().tolist(), [5] * 6)


class LoadTFRecordDatasetSequence(tf.test.TestCase):

  def _get_test_step(self,
                     global_step,
                     step_type,
                     dataset_spec):
    """Create step where all data (action, observation, etc) is global_step."""
    def _arr(shape):
      return np.full(shape, global_step, dtype=np.float32)

    observation = collections.OrderedDict()
    for key, value in dataset_spec.observation.items():
      observation[key] = _arr(value.shape)
    args = {
        'action': _arr(dataset_spec.action.shape),
        'observation': observation,
        'policy_info': (),
        'reward': _arr(shape=()),
        'discount': _arr(shape=()),
    }

    if step_type == StepType.FIRST:
      return trajectory.first(**args)
    elif step_type == StepType.MID:
      return trajectory.mid(**args)
    elif step_type == StepType.LAST:
      return trajectory.last(**args)

  def _init_test_shards(self,
                        num_shards,
                        episodes_per_shard,
                        steps_per_episode):
    """Build a test dataset of BlockPush data."""
    datadir = tempfile.mkdtemp(dir=self.get_temp_dir())
    shards = [os.path.join(datadir, 'shard%d' % i) for i in range(num_shards)]

    # Replicate the ibc data pattern to keep episodes within a single
    # shard (episodes never straddle shard boundary).
    # Initialize an environment and policy just to get the data spec.
    env = suite_gym.load('BlockPush-v0')
    policy = oriented_push_oracle_module.OrientedPushOracle(env)

    global_step = 0
    for shard in shards:
      observer = example_encoding_dataset.TFRecordObserver(
          shard, policy.collect_data_spec, py_mode=True)

      assert steps_per_episode > 2
      for _ in range(episodes_per_shard):
        for i_step in range(steps_per_episode):
          if i_step == 0:
            step_type = StepType.FIRST
          elif i_step == steps_per_episode - 1:
            step_type = StepType.LAST
          else:
            step_type = StepType.MID

          traj = self._get_test_step(
              global_step, step_type, policy.collect_data_spec)
          observer(traj)
          global_step += 1

    return shards

  def _check_sample(self, sample, expected_values, step_type, next_step_type):
    self.assertEqual(sample.step_type.numpy().tolist(), step_type)
    self.assertEqual(sample.next_step_type.numpy().tolist(), next_step_type)

    self.assertEqual(sample.reward.numpy().tolist(), expected_values)
    self.assertEqual(sample.discount.numpy().tolist(), expected_values)
    self.assertLen(sample.action.shape, 2)  # (seq_len, n)
    self.assertEqual(sample.action.numpy().tolist(),
                     [[v] * sample.action.shape[1] for v in expected_values])
    for value in sample.observation.values():
      self.assertLen(value.shape, 2)  # (seq_len, n)
      self.assertEqual(value.numpy().tolist(),
                       [[v] * value.shape[1] for v in expected_values])

  def test_load_tfrecord_dataset_sequence(self):
    shards = self._init_test_shards(num_shards=2,
                                    episodes_per_shard=2,
                                    steps_per_episode=4)
    dataset = load_tfrecord_dataset_sequence(shards, seq_len=3,
                                             deterministic=True)
    dataset_iter = iter(dataset)
    self._check_sample(next(dataset_iter), [0, 1, 2],  # 1st shard, 0
                       [StepType.FIRST, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [8, 9, 10],  # 2nd shard, 0
                       [StepType.FIRST, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [1, 2, 3],  # 1st shard, 1
                       [StepType.MID, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.LAST])
    self._check_sample(next(dataset_iter), [9, 10, 11],  # 2nd shard, 1
                       [StepType.MID, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.LAST])
    self._check_sample(next(dataset_iter), [4, 4, 4],  # 1st shard, 2
                       [StepType.FIRST, StepType.FIRST, StepType.FIRST],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [12, 12, 12],  # 2nd shard, 2
                       [StepType.FIRST, StepType.FIRST, StepType.FIRST],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [4, 4, 5],  # 1st shard, 3
                       [StepType.FIRST, StepType.FIRST, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [12, 12, 13],  # 2nd shard, 3
                       [StepType.FIRST, StepType.FIRST, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [4, 5, 6],  # 1st shard, 4
                       [StepType.FIRST, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [12, 13, 14],  # 2nd shard, 4
                       [StepType.FIRST, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [5, 6, 7],  # 1st shard, 5
                       [StepType.MID, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.LAST])
    self._check_sample(next(dataset_iter), [13, 14, 15],  # 2nd shard, 5
                       [StepType.MID, StepType.MID, StepType.MID],
                       [StepType.MID, StepType.MID, StepType.LAST])
    self._check_sample(next(dataset_iter), [0, 0, 0],  # 1st shard, 6
                       [StepType.FIRST, StepType.FIRST, StepType.FIRST],
                       [StepType.MID, StepType.MID, StepType.MID])
    self._check_sample(next(dataset_iter), [8, 8, 8],  # 2nd shard, 6
                       [StepType.FIRST, StepType.FIRST, StepType.FIRST],
                       [StepType.MID, StepType.MID, StepType.MID])


if __name__ == '__main__':
  tf.test.main()
