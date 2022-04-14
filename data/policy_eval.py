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

"""Evaluates TF-Agents policies."""
import functools
import os
import shutil

from absl import app
from absl import flags
from absl import logging

import gin
# Need import to get env resgistration.
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing import block_pushing_discontinuous
from ibc.environments.block_pushing import block_pushing_multimodal
from ibc.environments.collect.utils import get_oracle as get_oracle_module
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.environments.particle import particle_oracles
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.metrics import py_metrics
# Need import to get tensorflow_probability registration.
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import example_encoding_dataset


flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

flags.DEFINE_integer('num_episodes', 5, 'Number of episodes to evaluate.')
flags.DEFINE_integer('history_length', None,
                     'If set the previous n observations are stacked.')
flags.DEFINE_bool('video', False,
                  'If true record a video of the evaluations.')
flags.DEFINE_bool('viz_img', False,
                  'If true records an img of evaluation trajectories.')
flags.DEFINE_string('output_path', '/tmp/ibc/policy_eval/',
                    'Path to save videos at.')
flags.DEFINE_enum(
    'task', None,
    ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED',
     'PARTICLE', 'PUSH_DISCONTINUOUS', 'PUSH_MULTIMODAL'],
    'Which task of the enum to evaluate.')
flags.DEFINE_bool('use_image_obs', False,
                  'Whether to include image observations.')
flags.DEFINE_bool('flatten_env', False,
                  'If True the environment observations are flattened.')
flags.DEFINE_bool('shared_memory', False,
                  'If True the connection to pybullet uses shared memory.')
flags.DEFINE_string('saved_model_path', None,
                    'Path to the saved_model policy to eval.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to the checkpoint to evaluate.')
flags.DEFINE_enum('policy', None, [
    'random', 'oracle_reach', 'oracle_push', 'oracle_reach_normalized',
    'oracle_push_normalized', 'particle_green_then_blue'
], 'Static policies to evaluate.')
flags.DEFINE_string(
    'dataset_path', None,
    'If set a dataset of the policy evaluation will be saved '
    'to the given path.')
flags.DEFINE_integer('replicas', None,
                     'Number of parallel replicas generating evaluations.')


def evaluate(num_episodes,
             task,
             use_image_obs,
             shared_memory,
             flatten_env,
             saved_model_path=None,
             checkpoint_path=None,
             static_policy=None,
             dataset_path=None,
             history_length=None,
             video=False,
             viz_img=False,
             output_path=None):
  """Evaluates the given policy for n episodes."""
  if task in ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED']:
    # Options are supported through flags to build_env_name, and different
    # registered envs.
    env_name = block_pushing.build_env_name(task, shared_memory, use_image_obs)
  elif task in ['PUSH_DISCONTINUOUS']:
    env_name = block_pushing_discontinuous.build_env_name(
        task, shared_memory, use_image_obs)
  elif task in ['PUSH_MULTIMODAL']:
    env_name = block_pushing_multimodal.build_env_name(
        task, shared_memory, use_image_obs)
  elif task == 'PARTICLE':
    # Options are supported through gin, registered env is the same.
    env_name = 'Particle-v0'
    assert not (shared_memory or use_image_obs)  # Not supported.
  else:
    raise ValueError("I don't recognize this task to eval.")

  if flatten_env:
    env = suite_gym.load(
        env_name, env_wrappers=[wrappers.FlattenObservationsWrapper])
  else:
    env = suite_gym.load(env_name)

  if history_length:
    env = wrappers.HistoryWrapper(
        env, history_length=history_length, tile_first_step_obs=True)

  if video:
    video_path = output_path

    if saved_model_path:
      policy_name = os.path.basename(os.path.normpath(saved_model_path))
      checkpoint_ref = checkpoint_path.split('_')[-1]
      video_path = os.path.join(video_path,
                                policy_name + '_' + checkpoint_ref + 'vid.mp4')

    if static_policy:
      video_path = os.path.join(video_path, static_policy, 'vid.mp4')


  if saved_model_path and static_policy:
    raise ValueError(
        'Only pass in either a `saved_model_path` or a `static_policy`.')

  if saved_model_path:
    if not checkpoint_path:
      raise ValueError('Must provide a `checkpoint_path` with a saved_model.')
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        saved_model_path, load_specs_from_pbtxt=True)
    policy.update_from_checkpoint(checkpoint_path)
  else:
    if static_policy == 'random':
      policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                               env.action_spec())
    elif task == 'PARTICLE':
      if static_policy == 'particle_green_then_blue':
        # TODO(peteflorence): support more particle oracle options.
        policy = particle_oracles.ParticleOracle(env)
      else:
        raise ValueError('Unknown policy for given task: %s: ' % static_policy)
    elif task != 'PARTICLE':
      # Get an oracle.
      policy = get_oracle_module.get_oracle(env, flags.FLAGS.task)
    else:
      raise ValueError('Unknown policy: %s: ' % static_policy)

  metrics = [
      py_metrics.AverageReturnMetric(buffer_size=num_episodes),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes),
  ]
  env_metrics, success_metric = env.get_metrics(num_episodes)
  metrics += env_metrics

  observers = metrics[:]

  if viz_img and ('Particle' in env_name):
    visualization_dir = '/tmp/particle_oracle'
    shutil.rmtree(visualization_dir, ignore_errors=True)
    env.set_img_save_dir(visualization_dir)
    observers += [env.save_image]

  if dataset_path:
    # TODO(oars, peteflorence): Consider a custom observer to filter only
    # positive examples.
    observers.append(
        example_encoding_dataset.TFRecordObserver(
            dataset_path,
            policy.collect_data_spec,
            py_mode=True,
            compress_image=True))

  driver = py_driver.PyDriver(env, policy, observers, max_episodes=num_episodes)
  time_step = env.reset()
  initial_policy_state = policy.get_initial_state(1)
  driver.run(time_step, initial_policy_state)
  log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
  logging.info('\n\t\t '.join(log))

  env.close()


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.add_config_file_search_path(os.getcwd())
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                      flags.FLAGS.gin_bindings)

  if flags.FLAGS.replicas:
    jobs = []
    if not flags.FLAGS.dataset_path:
      raise ValueError(
          'A dataset_path must be provided when replicas are specified.')
    dataset_split_path = os.path.splitext(flags.FLAGS.dataset_path)
    context = multiprocessing.get_context()

    for i in range(flags.FLAGS.replicas):
      dataset_path = dataset_split_path[0] + '_%d' % i + dataset_split_path[1]
      kwargs = dict(
          num_episodes=flags.FLAGS.num_episodes,
          task=flags.FLAGS.task,
          use_image_obs=flags.FLAGS.use_image_obs,
          shared_memory=flags.FLAGS.shared_memory,
          flatten_env=flags.FLAGS.flatten_env,
          saved_model_path=flags.FLAGS.saved_model_path,
          checkpoint_path=flags.FLAGS.checkpoint_path,
          static_policy=flags.FLAGS.policy,
          dataset_path=dataset_path,
          history_length=flags.FLAGS.history_length
      )
      job = context.Process(target=evaluate, kwargs=kwargs)
      job.start()
      jobs.append(job)

    for job in jobs:
      job.join()

  else:
    evaluate(
        num_episodes=flags.FLAGS.num_episodes,
        task=flags.FLAGS.task,
        use_image_obs=flags.FLAGS.use_image_obs,
        shared_memory=flags.FLAGS.shared_memory,
        flatten_env=flags.FLAGS.flatten_env,
        saved_model_path=flags.FLAGS.saved_model_path,
        checkpoint_path=flags.FLAGS.checkpoint_path,
        static_policy=flags.FLAGS.policy,
        dataset_path=flags.FLAGS.dataset_path,
        history_length=flags.FLAGS.history_length,
        video=flags.FLAGS.video,
        viz_img=flags.FLAGS.viz_img,
        output_path=flags.FLAGS.output_path,
    )


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
