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

"""Main binary to train a Behavioral Cloning agent."""
#  pylint: disable=g-long-lambda
import collections
import datetime
import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing import block_pushing_discontinuous  # pylint: disable=unused-import
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.ibc import tasks
from ibc.ibc.agents import ibc_policy  # pylint: disable=unused-import
from ibc.ibc.eval import eval_env as eval_env_module
from ibc.ibc.train import get_agent as agent_module
from ibc.ibc.train import get_cloning_network as cloning_network_module
from ibc.ibc.train import get_data as data_module
from ibc.ibc.train import get_eval_actor as eval_actor_module
from ibc.ibc.train import get_learner as learner_module
from ibc.ibc.train import get_normalizers as normalizers_module
from ibc.ibc.train import get_sampling_spec as sampling_spec_module
from ibc.ibc.utils import make_video as video_module
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
import wandb

flags.DEFINE_string('tag', None,
                    'Tag for the experiment. Appended to the root_dir.')
flags.DEFINE_bool('add_time', False,
                  'If True current time is added to the experiment path.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_bool('shared_memory_eval', False,
                  'If true the eval_env uses shared_memory.')
flags.DEFINE_bool('video', False,
                  'If true, write out one rollout video after eval.')
flags.DEFINE_multi_enum(
    'task', None,
    (tasks.IBC_TASKS + tasks.D4RL_TASKS),
    'If True the reach task is evaluated.')
flags.DEFINE_boolean('viz_img', default=False,
                     help='Whether to save out imgs of what happened.')
flags.DEFINE_bool('skip_eval', False,
                  'If true the evals are skipped and instead run from '
                  'policy_eval binary.')
flags.DEFINE_bool('multi_gpu', False,
                  'If true, run in multi-gpu setting.')

flags.DEFINE_enum('device_type', 'gpu', ['gpu', 'tpu'],
                  'Where to perform training.')

FLAGS = flags.FLAGS
VIZIER_KEY = 'success'


@gin.configurable
def train_eval(
    task=None,
    dataset_path=None,
    root_dir=None,
    # 'ebm' or 'mse' or 'mdn'.
    loss_type=None,
    # Name of network to train. see get_cloning_network.
    network=None,
    # Training params
    batch_size=512,
    num_iterations=20000,
    learning_rate=1e-3,
    decay_steps=100,
    replay_capacity=100000,
    eval_interval=1000,
    eval_loss_interval=100,
    eval_episodes=1,
    fused_train_steps=100,
    sequence_length=2,
    uniform_boundary_buffer=0.05,
    for_rnn=False,
    flatten_action=True,
    dataset_eval_fraction=0.0,
    goal_tolerance=0.02,
    tag=None,
    add_time=False,
    seed=0,
    viz_img=False,
    skip_eval=False,
    num_envs=1,
    shared_memory_eval=False,
    image_obs=False,
    strategy=None,
    # Use this to sweep amount of tfrecords going into training.
    # -1 for 'use all'.
    max_data_shards=-1,
    use_warmup=False):
  """Trains a BC agent on the given datasets."""
  if task is None:
    raise ValueError('task argument must be set.')
  logging.info(('Using task:', task))

  tf.random.set_seed(seed)
  if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)

  # Logging.
  if tag:
    root_dir = os.path.join(root_dir, tag)
  if add_time:
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(root_dir, current_time)

  # Define eval env.
  eval_envs = []
  env_names = []
  for task_id in task:
    env_name = eval_env_module.get_env_name(task_id, shared_memory_eval,
                                            image_obs)
    logging.info(('Got env name:', env_name))
    eval_env = eval_env_module.get_eval_env(
        env_name, sequence_length, goal_tolerance, num_envs)
    logging.info(('Got eval_env:', eval_env))
    eval_envs.append(eval_env)
    env_names.append(env_name)

  obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(eval_envs[0]))

  # Compute normalization info from training data.
  create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
      dataset_path,
      sequence_length,
      replay_capacity,
      batch_size,
      for_rnn,
      dataset_eval_fraction,
      flatten_action)
  train_data, _ = create_train_and_eval_fns_unnormalized()
  (norm_info, norm_train_data_fn) = normalizers_module.get_normalizers(
      train_data, batch_size, env_name)

  # Create normalized training data.
  if not strategy:
    strategy = tf.distribute.get_strategy()
  per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
  create_train_and_eval_fns = data_module.get_data_fns(
      dataset_path,
      sequence_length,
      replay_capacity,
      per_replica_batch_size,
      for_rnn,
      dataset_eval_fraction,
      flatten_action,
      norm_function=norm_train_data_fn,
      max_data_shards=max_data_shards)
  # Create properly distributed eval data iterator.
  dist_eval_data_iter = get_distributed_eval_data(create_train_and_eval_fns,
                                                  strategy)

  # Create normalization layers for obs and action.
  with strategy.scope():
    # Create train step counter.
    train_step = train_utils.create_train_step()

    # Define action sampling spec.
    action_sampling_spec = sampling_spec_module.get_sampling_spec(
        action_tensor_spec,
        min_actions=norm_info.min_actions,
        max_actions=norm_info.max_actions,
        uniform_boundary_buffer=uniform_boundary_buffer,
        act_norm_layer=norm_info.act_norm_layer)

    # This is a common opportunity for a bug, having the wrong sampling min/max
    # so log this.
    logging.info(('Using action_sampling_spec:', action_sampling_spec))

    # Define keras cloning network.
    cloning_network = cloning_network_module.get_cloning_network(
        network,
        obs_tensor_spec,
        action_tensor_spec,
        norm_info.obs_norm_layer,
        norm_info.act_norm_layer,
        sequence_length,
        norm_info.act_denorm_layer)

    # Define tfagent.
    agent = agent_module.get_agent(loss_type,
                                   time_step_tensor_spec,
                                   action_tensor_spec,
                                   action_sampling_spec,
                                   norm_info.obs_norm_layer,
                                   norm_info.act_norm_layer,
                                   norm_info.act_denorm_layer,
                                   learning_rate,
                                   use_warmup,
                                   cloning_network,
                                   train_step,
                                   decay_steps)

    # Define bc learner.
    bc_learner = learner_module.get_learner(
        loss_type,
        root_dir,
        agent,
        train_step,
        create_train_and_eval_fns,
        fused_train_steps,
        strategy)

    # Define eval.
    eval_actors, eval_success_metrics = [], []
    for eval_env, env_name in zip(eval_envs, env_names):
      env_name_clean = env_name.replace('/', '_')
      eval_actor, success_metric = eval_actor_module.get_eval_actor(
          agent,
          env_name,
          eval_env,
          train_step,
          eval_episodes,
          root_dir,
          viz_img,
          num_envs,
          strategy,
          summary_dir_suffix=env_name_clean)
      eval_actors.append(eval_actor)
      eval_success_metrics.append(success_metric)

    get_eval_loss = tf.function(agent.get_eval_loss)

    # Get summary writer for aggregated metrics.
    aggregated_summary_dir = os.path.join(root_dir, 'eval')
    summary_writer = tf.summary.create_file_writer(
        aggregated_summary_dir, flush_millis=10000)
  logging.info('Saving operative-gin-config.')
  with tf.io.gfile.GFile(
      os.path.join(root_dir, 'operative-gin-config.txt'), 'wb') as f:
    f.write(gin.operative_config_str())

  # Main train and eval loop.
  while train_step.numpy() < num_iterations:
    # Run bc_learner for fused_train_steps.
    training_step(agent, bc_learner, fused_train_steps, train_step)

    if (dist_eval_data_iter is not None and
        train_step.numpy() % eval_loss_interval == 0):
      # Run a validation step.
      validation_step(
          dist_eval_data_iter, bc_learner, train_step, get_eval_loss)

    if not skip_eval and train_step.numpy() % eval_interval == 0:

      all_metrics = []
      for eval_env, eval_actor, env_name, success_metric in zip(
          eval_envs, eval_actors, env_names, eval_success_metrics):
        # Run evaluation.
        metrics = evaluation_step(
            eval_episodes,
            eval_env,
            eval_actor,
            name_scope_suffix=f'_{env_name}')
        all_metrics.append(metrics)

        # rendering on some of these envs is broken
        if FLAGS.video and 'kitchen' not in task:
          if 'PARTICLE' in task:
            # A seed with spread-out goals is more clear to visualize.
            eval_env.seed(42)
          # Write one eval video.
          video_module.make_video(
              agent,
              eval_env,
              root_dir,
              step=train_step.numpy(),
              strategy=strategy)

      metric_results = collections.defaultdict(list)
      for env_metrics in all_metrics:
        for metric in env_metrics:
          metric_results[metric.name].append(metric.result())

      with summary_writer.as_default(), \
         common.soft_device_placement(), \
         tf.summary.record_if(lambda: True):
        for key, value in metric_results.items():
          tf.summary.scalar(
              name=os.path.join('AggregatedMetrics/', key),
              data=sum(value) / len(value),
              step=train_step)

  summary_writer.flush()


def training_step(agent, bc_learner, fused_train_steps, train_step):
  """Runs bc_learner for fused training steps."""
  reduced_loss_info = None
  if not hasattr(agent, 'ebm_loss_type') or agent.ebm_loss_type != 'cd_kl':
    reduced_loss_info = bc_learner.run(iterations=fused_train_steps)
  else:
    for _ in range(fused_train_steps):
      # I think impossible to do this inside tf.function.
      agent.cloning_network_copy.set_weights(
          agent.cloning_network.get_weights())
      reduced_loss_info = bc_learner.run(iterations=1)

  if reduced_loss_info:
    # Graph the loss to compare losses at the same scale regardless of
    # number of devices used.
    with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
        True):
      tf.summary.scalar(
          'reduced_loss', reduced_loss_info.loss, step=train_step)


def validation_step(dist_eval_data_iter, bc_learner, train_step,
                    get_eval_loss_fn):
  """Runs a validation step."""
  losses_dict = get_eval_loss_fn(next(dist_eval_data_iter))

  with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
      True):
    common.summarize_scalar_dict(
        losses_dict, step=train_step, name_scope='Eval_Losses/')


def evaluation_step(eval_episodes, eval_env, eval_actor, name_scope_suffix=''):
  """Evaluates the agent in the environment."""
  logging.info('Evaluating policy.')
  with tf.name_scope('eval' + name_scope_suffix):
    # This will eval on seeds:
    # [0, 1, ..., eval_episodes-1]
    for eval_seed in range(eval_episodes):
      eval_env.seed(eval_seed)
      eval_actor.reset()  # With the new seed, the env actually needs reset.
      eval_actor.run()

    eval_actor.log_metrics()
    eval_actor.write_metric_summaries()
  return eval_actor.metrics


def get_distributed_eval_data(data_fn, strategy):
  """Gets a properly distributed evaluation data iterator."""
  _, eval_data = data_fn()
  dist_eval_data_iter = None
  if eval_data:
    dist_eval_data_iter = iter(
        strategy.distribute_datasets_from_function(lambda: eval_data))
  return dist_eval_data_iter


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.add_config_file_search_path(os.getcwd())
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings,
                                      # TODO(coreylynch): This is a temporary
                                      # hack until we get proper distributed
                                      # eval working. Remove it once we do.
                                      skip_unknown=True)

  wandb.init(
    project="google-research-ibc",
    sync_tensorboard=True,
  )
  # Print operative gin config to stdout so wandb can intercept.
  # (it'd be nice for gin to provide a flat/nested dictionary of values so they
  # can be used via wandb's aggregation...)
  print(gin.config.config_str())

  # For TPU, FLAGS.tpu will be set with a TPU address and FLAGS.use_gpu
  # will be False.
  # For GPU, FLAGS.tpu will be None and FLAGS.use_gpu will be True.
  strategy = strategy_utils.get_strategy(
      tpu=FLAGS.tpu, use_gpu=FLAGS.use_gpu)

  task = FLAGS.task or gin.REQUIRED
  # If setting this to True, change `my_rangea in mcmc.py to `= range`
  tf.config.experimental_run_functions_eagerly(False)

  train_eval(
      task=task,
      tag=FLAGS.tag,
      add_time=FLAGS.add_time,
      viz_img=FLAGS.viz_img,
      skip_eval=FLAGS.skip_eval,
      shared_memory_eval=FLAGS.shared_memory_eval,
      strategy=strategy)


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
