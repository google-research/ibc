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

"""Utils for serializing data as pybullet state or x100 training tfrecords."""
import collections
import os
from typing import List

from ibc.environments.utils.utils_pybullet import write_pybullet_state
import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import trajectory
from tf_agents.utils import example_encoding_dataset

EpisodeData = collections.namedtuple('EpisodeData',
                                     ('time_step', 'action', 'pybullet_state'))


def get_tfrecord_observers(env,
                           tfagents_path=None,
                           worker_id=None,
                           dataset_nshards=None):
  """Initialize TFAgent observers (for saving tfrecord data). One per shard."""
  observers = []
  if tfagents_path:
    tf.io.gfile.makedirs(os.path.dirname(tfagents_path))
    observers = _init_observers(env, tfagents_path, worker_id, dataset_nshards)
  return observers


def _init_observers(env, tfagents_path, worker_id, dataset_nshards):
  """Initialize TFAgent observers (for saving data). One per shard."""
  observers = []
  for i in range(dataset_nshards):
    dataset_split_path = os.path.splitext(tfagents_path)
    data_spec = trajectory.from_transition(
        env.time_step_spec(), policy_step.PolicyStep(env.action_spec()),
        env.time_step_spec())
    observers.append(
        example_encoding_dataset.TFRecordObserver(
            dataset_split_path[0] + '_%d' % i + '_worker-%d' % worker_id +
            dataset_split_path[1],
            data_spec,
            py_mode=True,
            compress_image=True))
  return observers


def write_tfagents_data(
    episode_data,
    observer):
  """Write out the episode data using the TFRecordObserver."""
  assert episode_data.action, 'empty episode (got done on reset)'
  assert len(episode_data.action) + 1 == len(episode_data.time_step)
  for i in range(len(episode_data.action)):
    traj = trajectory.from_transition(episode_data.time_step[i],
                                      episode_data.action[i],
                                      episode_data.time_step[i + 1])
    observer(traj)


def write_pybullet_data(task,
                        pybullet_state_path,
                        episode_data,
                        i_episode,
                        worker_id):
  """Write out the pybullet json for replay."""
  actions = [policy_step.action.tolist() for policy_step in episode_data.action]
  filename = '%s_ep%d_worker-%d.json.zip' % (pybullet_state_path, i_episode,
                                             worker_id)
  write_pybullet_state(
      filename, episode_data.pybullet_state, task, actions=actions)
