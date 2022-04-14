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

r"""Binary to perform oracle data collection.

Note that in the above command TFAgent tfrecords and the episode json contain
redundant data. In most applications you would store one or the other.
"""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from ibc.environments.block_pushing import block_pushing
from ibc.environments.collect.utils import get_env as get_env_module
from ibc.environments.collect.utils import get_oracle as get_oracle_module
from ibc.environments.collect.utils import serialize as serialize_module
import numpy as np  # pylint: disable=unused-import,g-bad-import-order
import tensorflow as tf
from tf_agents.environments import suite_gym  # pylint: disable=unused-import,g-bad-import-order
from tf_agents.google.wrappers import mp4_video_wrapper
from tf_agents.trajectories import policy_step

flags.DEFINE_enum(
    'task',
    None,
    block_pushing.BlockTaskVariant._member_names_,  # pylint: disable=protected-access
    'Which task to run')
flags.DEFINE_bool('use_image_obs', True,
                  'Whether to include image observations.')
flags.DEFINE_bool('fixed_start_poses', False, 'Whether to use fixed start '
                  'poses.')
flags.DEFINE_bool('noisy_ee_pose', False, 'Whether to use noisy pose '
                  'for end effector so it does not start in exact position.')
flags.DEFINE_bool('no_episode_step_limit', False,
                  'If True, remove max_episode_steps step limit.')
flags.DEFINE_string(
    'tfagents_path', None,
    'If set a dataset of the oracle output will be saved '
    'to the given path.')
flags.DEFINE_integer('dataset_nshards', 1, 'Number of dataset shards to save.')
flags.DEFINE_string(
    'pybullet_state_path', None,
    'If set a json record of full pybullet, action and state '
    'will be saved to the given path.')
flags.DEFINE_bool('shared_memory', False, 'Shared memory for pybullet.')
flags.DEFINE_bool('save_successes_only', True,
                  'Whether to save only successful episodes.')
flags.DEFINE_integer('num_episodes', 1,
                     'The number of episodes to collect.')
flags.DEFINE_integer('worker_id', 0, 'Worker id of the replica.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas')
flags.DEFINE_bool('video', False,
                  'If true record a video of the evaluations.')
flags.DEFINE_string('video_path', None, 'Path to save videos at.')

FLAGS = flags.FLAGS
flags.mark_flags_as_required(['task'])

MAX_EPISODE_RECORD_STEPS = 10_000


def main(argv):
  del argv

  # Load an environment for this task with no step limit.
  env = get_env_module.get_env(
      FLAGS.task,
      use_image_obs=FLAGS.use_image_obs,
      fixed_start_poses=FLAGS.fixed_start_poses,
      noisy_ee_pose=FLAGS.noisy_ee_pose,
      max_episode_steps=np.inf if FLAGS.no_episode_step_limit else None)

  if FLAGS.video:
    # Write debug videos to video_path.
    env = mp4_video_wrapper.Mp4VideoWrapper(
        env, env.control_frequency,
        frame_interval=1, video_filepath=FLAGS.video_path)

  # Get an oracle.
  oracle_policy = get_oracle_module.get_oracle(env, FLAGS.task)

  # env resets are done via directly restoring pybullet state. Update
  # internal state now that we've added additional visual objects.
  if hasattr(env, 'save_state'):
    env.save_state()

  # If tfagents_path specified, create tfrecord observers for writing out
  # training data.
  observers = serialize_module.get_tfrecord_observers(
      env,
      tfagents_path=FLAGS.tfagents_path,
      worker_id=FLAGS.worker_id,
      dataset_nshards=FLAGS.dataset_nshards)

  if FLAGS.pybullet_state_path:
    tf.io.gfile.makedirs(os.path.dirname(FLAGS.pybullet_state_path))

  cur_observer = 0
  num_episodes = 0
  num_failures = 0
  total_num_steps = 0

  while True:
    logging.info('Starting episode %d.', num_episodes)
    episode_data = serialize_module.EpisodeData(
        time_step=[], action=[], pybullet_state=[])

    time_step = env.reset()
    episode_data.time_step.append(time_step)
    episode_data.pybullet_state.append(env.get_pybullet_state())

    if hasattr(env, 'instruction') and env.instruction is not None:
      logging.info('Current instruction: %s',
                   env.decode_instruction(env.instruction))

    done = time_step.is_last()
    reward = 0.0

    if 'instruction' in time_step.observation:
      instruction = time_step.observation['instruction']
      nonzero_ints = instruction[instruction != 0]
      nonzero_bytes = bytes(nonzero_ints)
      clean_text = nonzero_bytes.decode('utf-8')
      logging.info(clean_text)

    while not done:
      action = oracle_policy.action(time_step,
                                    oracle_policy.get_initial_state(1)).action

      time_step = env.step(action)

      if len(episode_data.action) < MAX_EPISODE_RECORD_STEPS:
        episode_data.action.append(
            policy_step.PolicyStep(action=action, state=(), info=()))
        episode_data.time_step.append(time_step)
        episode_data.pybullet_state.append(env.get_pybullet_state())

      done = time_step.is_last()
      reward = time_step.reward

    if done:  # episode terminated normally (not on manual reset w.o. saving).
      # Skip saving if it didn't end in success.
      if FLAGS.save_successes_only and reward <= 0:
        logging.info('Skipping episode that did not end in success.')
        num_failures += 1
        continue

      num_episodes += 1

      if observers:
        total_num_steps += len(episode_data.action)
        logging.info('Recording %d length episode to shard %d',
                     len(episode_data.action), cur_observer)
        serialize_module.write_tfagents_data(
            episode_data, observers[cur_observer])
        cur_observer = (cur_observer + 1) % len(observers)

      if FLAGS.pybullet_state_path is not None:
        serialize_module.write_pybullet_data(FLAGS.task,
                                             FLAGS.pybullet_state_path,
                                             episode_data,
                                             num_episodes,
                                             FLAGS.worker_id)

      if num_episodes >= FLAGS.num_episodes:
        logging.info(
            'Num episodes: %d Num failures: %d', num_episodes, num_failures)
        logging.info('Avg steps: %f', total_num_steps / num_episodes)

        return


if __name__ == '__main__':
  app.run(main)
