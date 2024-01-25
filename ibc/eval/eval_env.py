# coding=utf-8
# Copyright 2024 The Reach ML Authors.
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

"""Loads an eval environment based on task name."""
import functools

from absl import flags

from ibc.environments.block_pushing import block_pushing
from ibc.environments.block_pushing import block_pushing_discontinuous
from ibc.environments.block_pushing import block_pushing_multimodal
from ibc.ibc import tasks
try:
  from ibc.ibc.eval import d4rl_utils  # pylint: disable=g-import-not-at-top
except:  # pylint: disable=bare-except
  print('WARNING: Could not import d4rl.')
from tf_agents.environments import parallel_py_environment  # pylint: disable=g-import-not-at-top
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers

flags.DEFINE_bool('eval_env_2dboard_use_normalized_env', False,
                  'If true, load the normalized version of the environment '
                  '(requires different demonstration tfrecords).')

FLAGS = flags.FLAGS


def get_env_name(task, shared_memory_eval, use_image_obs=False):
  """Returns environment name for a given task."""
  if task in ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED']:
    env_name = block_pushing.build_env_name(
        task, shared_memory_eval, use_image_obs=use_image_obs)
  elif task in ['PUSH_DISCONTINUOUS']:
    env_name = block_pushing_discontinuous.build_env_name(
        task, shared_memory_eval, use_image_obs=use_image_obs)
  elif task in ['PUSH_MULTIMODAL']:
    env_name = block_pushing_multimodal.build_env_name(
        task, shared_memory_eval, use_image_obs=use_image_obs)
  elif task == 'PARTICLE':
    env_name = 'Particle-v0'
    assert not shared_memory_eval  # Not supported.
    assert not use_image_obs  # Not supported.
  elif task in tasks.D4RL_TASKS:
    env_name = task
    assert not use_image_obs  # Not supported.
  else:
    raise ValueError('unknown task %s' % task)
  return env_name


def get_eval_env(env_name, sequence_length, goal_tolerance, num_envs):
  """Returns an eval environment for the given task."""
  if env_name in tasks.D4RL_TASKS:
    load_env_fn = d4rl_utils.load_d4rl
  else:
    load_env_fn = suite_gym.load

  if num_envs > 1:
    def load_env_and_wrap(env_name):
      eval_env = load_env_fn(env_name)
      eval_env = wrappers.HistoryWrapper(
          eval_env, history_length=sequence_length, tile_first_step_obs=True)
      return eval_env

    env_ctor = functools.partial(load_env_and_wrap, env_name)
    eval_env = parallel_py_environment.ParallelPyEnvironment(
        [env_ctor] * num_envs, start_serially=False)
  else:
    eval_env = load_env_fn(env_name)
    if env_name not in tasks.D4RL_TASKS and 'Block' in env_name:
      eval_env.set_goal_dist_tolerance(goal_tolerance)
    eval_env = wrappers.HistoryWrapper(
        eval_env, history_length=sequence_length, tile_first_step_obs=True)

  return eval_env
