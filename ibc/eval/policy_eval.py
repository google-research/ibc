# coding=utf-8
# Copyright 2021 The Reach ML Authors.
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

"""Parallel policy eval for D4RL environment."""
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
from ibc.ibc import tasks
from ibc.ibc.agents import ibc_policy
from ibc.ibc.eval import eval_env as eval_env_module
# Need import to get tensorflow_probability registration.
import tensorflow as tf
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.train import actor

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

flags.DEFINE_enum('task', None,
                  (tasks.IBC_TASKS +
                   tasks.D4RL_TASKS),
                  'Task to evaluate on.')
flags.DEFINE_integer('replica_id', None,
                     'The replica ID from 0 to num_replicas-1.')
flags.DEFINE_bool('shared_memory_eval', False,
                  'If true the eval_env uses shared_memory.')
flags.DEFINE_bool('late_fusion', False,
                  'If true the policy is configured with late fusion.')

FLAGS = flags.FLAGS

EVALUATED_STEPS_FILE = 'evaluated_steps.txt'


def get_already_evaluated_checkpoints(eval_log_dir):
  """Returns the evaluated_file path and the set of evaluated checkpoints."""
  evaluated_file = os.path.join(eval_log_dir, EVALUATED_STEPS_FILE)
  evaluated_checkpoints = set()

  # Load already evaluated checkpoints.
  if tf.io.gfile.exists(evaluated_file):
    with tf.io.gfile.GFile(evaluated_file, 'r') as f:
      for step in f:
        evaluated_checkpoints.add(step.strip())

  return evaluated_file, evaluated_checkpoints


def get_checkpoints_to_evaluate(evaluated_checkpoints, saved_model_dir):
  """Gets a list of checkpoints to evaluate."""
  while True:
    checkpoints_dir = os.path.join(saved_model_dir, 'checkpoints', '*')
    checkpoints = tf.io.gfile.glob(checkpoints_dir)
    if not checkpoints:
      time.sleep(30)
      logging.info('No checkpoint yet, sleeping for 30 seconds.')
    else:
      if len(tf.io.gfile.listdir(os.path.join(
          checkpoints[0], 'variables'))) == 2:
        # Only break when the first checkpoint has been fully written. Trying
        # to restore partial writes results in runtime errors.
        break
  return sorted(list(set(checkpoints) - evaluated_checkpoints), reverse=True)


def build_policy(saved_model_path,
                 checkpoint_path=None,
                 seconds_between_checkpoint_polls=10,
                 late_fusion=False):
  """Builds a policy using the saved_model."""
  # Load saved model.
  while not (
      tf.io.gfile.exists(os.path.join(saved_model_path, 'saved_model.pb')) and
      tf.io.gfile.exists(os.path.join(saved_model_path, 'policy_specs.pbtxt'))):
    logging.info('Waiting on the first checkpoint to become available.')
    time.sleep(seconds_between_checkpoint_polls)

  policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
      saved_model_path, load_specs_from_pbtxt=True)
  if checkpoint_path:
    policy.update_from_checkpoint(checkpoint_path)

  # Here we could use the loaded policy directly, however in order to allow us
  # to change DFO parameters we create a new instance that we can modify with
  # gin bindings or configs.
  time_step_tensor_spec = tensor_spec.from_spec(policy.time_step_spec)
  action_tensor_spec = tensor_spec.from_spec(policy.action_spec)

  try:
    sampling_minimum = [
        v for v in policy.variables() if 'sampling/minimum' in v.name
    ][0].numpy()
    sampling_maximum = [
        v for v in policy.variables() if 'sampling/maximum' in v.name
    ][0].numpy()
  except IndexError:
    raise ValueError('Policy did not save sampling bounds.')

  action_sampling_spec = tensor_spec.BoundedTensorSpec(
      shape=action_tensor_spec.shape,
      dtype=action_tensor_spec.dtype,
      minimum=sampling_minimum,
      maximum=sampling_maximum)

  logging.info('action_sampling_spec: %r', action_sampling_spec)
  new_policy = ibc_policy.IbcPolicy(
      time_step_tensor_spec,
      action_tensor_spec,
      action_sampling_spec,
      policy.cloning_network,
      late_fusion=late_fusion)
  new_py_policy = py_tf_eager_policy.PyTFEagerPolicy(new_policy)
  # Monkey patch the action method to use the modified DFO.
  policy.action = new_py_policy.action

  return policy


def save_operative_gin_config(eval_log_dir):
  logging.info('Saving operative-gin-config.')
  if not tf.io.gfile.exists(eval_log_dir):
    tf.io.gfile.makedirs(eval_log_dir)
  with tf.io.gfile.GFile(
      os.path.join(eval_log_dir, 'operative-gin-config.txt'), 'wb') as f:
    f.write(gin.operative_config_str())


@gin.configurable
def continuous_eval(env_name,
                    experiment_dir=None,
                    eval_log_dir=None,
                    num_episodes=None,
                    task=None,
                    sequence_length=2,
                    goal_tolerance=0.02,
                    max_train_step=100000,
                    late_fusion=False,):
  """Evaluates a checkpoint directory.

  Checkpoints for the saved model to evaluate are assumed to be at the same
  directory level as the saved_model dir. ie:

  * saved_model_dir: root_dir/policies/greedy_policy
  * checkpoints_dir: root_dir/policies/checkpoints

  Args:
    env_name: Str, name of eval environment.
    experiment_dir: root dir for checkpoints.
    eval_log_dir: Optional path to output summaries of the evaluations. If None
      no summaries are written.
    num_episodes: Number or episodes to evaluate per checkpoint.
    task: Str task/environment name, e.g. "hopper-expert-v0".
    sequence_length: How much history is added to each observation at training
      time.
    goal_tolerance: float, tolerance for block environments.
    max_train_step: Maximum train_step to evaluate. Once a train_step greater or
      equal to this is evaluated the evaluations will terminate.
    late_fusion: If True, observation tiling must be cone in the
      actor_network to match the action rather than in the policy.
  """
  eval_log_dir = os.path.join(eval_log_dir, task)
  if FLAGS.replica_id is not None:
    eval_log_dir = os.path.join(eval_log_dir, str(FLAGS.replica_id))

  saved_model_path = os.path.join(experiment_dir, 'policies', 'greedy_policy')

  policy = build_policy(saved_model_path, late_fusion=late_fusion)
  # Define eval env.
  env = eval_env_module.get_eval_env(env_name, sequence_length, goal_tolerance,
                                     num_envs=1)
  split = os.path.split(saved_model_path)
  # Remove trailing slash if we have one.
  if not split[-1]:
    policy_dir = split[0]
  else:
    policy_dir = saved_model_path
  policy_dir = os.path.dirname(policy_dir)

  if max_train_step and policy.get_train_step() > max_train_step:
    logging.info(
        'Policy train_step (%d) > max_train_step (%d). '
        'No evaluations performed.', policy.get_train_step(), max_train_step)
    return

  evaluated_file, evaluated_checkpoints = get_already_evaluated_checkpoints(
      eval_log_dir)
  checkpoint_list = get_checkpoints_to_evaluate(evaluated_checkpoints,
                                                policy_dir)

  last_eval_step = policy.get_train_step()

  eval_actor = actor.Actor(
      env,
      policy,
      tf.Variable(policy.get_train_step()),
      # Uncomment to see logs and measure how long steps take.
      observers=[lambda _: logging.info('step')],
      metrics=actor.eval_metrics(num_episodes),
      summary_dir=os.path.join(eval_log_dir, 'eval'),
      episodes_per_run=num_episodes)

  save_operative_gin_config(eval_log_dir)

  continuous = True
  with tf.io.gfile.GFile(evaluated_file, 'a') as f:
    while checkpoint_list or (continuous and last_eval_step < max_train_step):
      if not checkpoint_list and continuous:
        logging.info('Waiting on new checkpoints to become available.')
        time.sleep(5)
        checkpoint_list = get_checkpoints_to_evaluate(evaluated_checkpoints,
                                                      policy_dir)
      checkpoint = checkpoint_list.pop()
      policy.update_from_checkpoint(checkpoint)
      logging.info('Evaluating:\n\tStep:%d\tcheckpoint: %s',
                   policy.get_train_step(), checkpoint)

      eval_actor.train_step.assign(policy.get_train_step())
      eval_actor.run_and_log()
      f.write(checkpoint + '\n')
      f.flush()
      last_eval_step = policy.get_train_step()


def main(_):
  gin.add_config_file_search_path(os.getcwd())
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings,
                                      # TODO(coreylynch): This is a temporary
                                      # hack until we get proper distributed
                                      # eval working. Remove it once we do.
                                      skip_unknown=True)
  tf.config.experimental_run_functions_eagerly(False)
  task = FLAGS.task or gin.REQUIRED
  env_name = eval_env_module.get_env_name(task, FLAGS.shared_memory_eval)
  continuous_eval(env_name, task=FLAGS.task, late_fusion=FLAGS.late_fusion)

if __name__ == '__main__':
  app.run(main)
