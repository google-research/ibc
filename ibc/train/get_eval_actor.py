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

"""Defines an eval actor."""
import os

from absl import logging
from ibc.environments.d4rl import metrics as d4rl_metrics
from ibc.ibc import tasks
from ibc.ibc.utils import strategy_policy
from tf_agents.train import actor


def log_episode_complete(traj):
  if traj.is_boundary():
    logging.info('Completed episode.')


def get_eval_actor(agent,
                   env_name,
                   eval_env,
                   train_step,
                   eval_episodes,
                   root_dir,
                   viz_img,
                   num_envs,
                   strategy,
                   summary_dir_suffix=''):
  """Defines eval actor."""
  if num_envs > 1:
    eval_greedy_policy = agent.policy
  else:
    eval_greedy_policy = strategy_policy.StrategyPyTFEagerPolicy(
        agent.policy, strategy=strategy)

  metrics = actor.eval_metrics(eval_episodes)
  if env_name in tasks.D4RL_TASKS:
    success_metric = metrics[0]
    if env_name in tasks.ADROIT_TASKS:
      # Define custom eval success metric for Adroit tasks, since the rewards
      # include reward shaping terms.
      metrics += [
          d4rl_metrics.D4RLSuccessMetric(
              env=eval_env, buffer_size=eval_episodes)
      ]
  else:
    env_metrics, success_metric = eval_env.get_metrics(eval_episodes)
    metrics += env_metrics

  summary_dir = os.path.join(root_dir, 'eval', summary_dir_suffix)

  observers = []
  # Adds a log when an episode is done, allows seeing eval time in the logs.
  observers += [log_episode_complete]

  if viz_img and 'Particle' in env_name:
    eval_env.set_img_save_dir(summary_dir)
    observers += [eval_env.save_image]

  eval_actor = actor.Actor(
      eval_env,
      eval_greedy_policy,
      train_step,
      observers=observers,
      metrics=metrics,
      summary_dir=summary_dir,
      episodes_per_run=1,  # we are doing seeding, need to handle ourselves.
      summary_interval=-1)  # -1 will make so never automatically writes.
  return eval_actor, success_metric
