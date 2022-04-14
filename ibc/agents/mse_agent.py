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

"""MSE BC agent."""

from typing import Optional, Text

import gin
from ibc.ibc.agents import base_agent
from ibc.ibc.agents import mse_policy
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common


@gin.configurable
class MseBehavioralCloningAgent(base_agent.BehavioralCloningAgent):
  """Implements Mean Square Error-based behavior cloning."""

  def __init__(self,
               time_step_spec,
               action_spec,
               action_sampling_spec,
               cloning_network,
               optimizer,
               obs_norm_layer=None,
               act_norm_layer=None,
               act_denorm_layer=None,
               debug_summaries = False,
               summarize_grads_and_vars = False,
               train_step_counter = None,
               name = None):
    # tf.Module dependency allows us to capture checkpints and saved models with
    # the agent.
    tf.Module.__init__(self, name=name)

    self._action_sampling_spec = action_sampling_spec
    self._obs_norm_layer = obs_norm_layer
    self._act_norm_layer = act_norm_layer
    self._act_denorm_layer = act_denorm_layer
    self.cloning_network = cloning_network
    self.cloning_network.create_variables(training=False)

    self._optimizer = optimizer

    # Collect policy would normally be used for data collection. In a BCAgent
    # we don't expect to use it, unless we want to upgrade this to a DAGGER like
    # setup.

    collect_policy = mse_policy.MsePolicyWrapper(time_step_spec, action_spec,
                                                 cloning_network,
                                                 self._obs_norm_layer)
    policy = greedy_policy.GreedyPolicy(collect_policy)

    super(MseBehavioralCloningAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _loss(self,
            experience,
            variables_to_train=None,
            weights = None,
            training = False):
    observations, actions = experience

    # Use first observation to figure out batch/time sizes as they should be the
    # same across all observations.
    single_obs = tf.nest.flatten(observations)[0]
    batch_size = tf.shape(single_obs)[0]

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(variables_to_train)
      with tf.name_scope('loss'):
        network_state = self.cloning_network.get_initial_state(batch_size)

        bc_output, _ = self.cloning_network(
            observations, training=training, network_state=network_state)

        if isinstance(bc_output, tfp.distributions.Distribution):
          bc_action = bc_output.sample()
        else:
          bc_action = bc_output

        losses = tf.nest.map_structure(tf.losses.mse, actions, bc_action)
        # Flatten and add_n across all actions.
        losses = tf.nest.flatten(losses)
        per_example_loss = tf.add_n(losses)

        agg_loss = common.aggregate_losses(
            per_example_loss=per_example_loss,
            sample_weight=weights,
            regularization_loss=self.cloning_network.losses)

        total_loss = agg_loss.total_loss

        losses_dict = {'mse_total_loss': total_loss}

        common.summarize_scalar_dict(
            losses_dict, step=self.train_step_counter, name_scope='Losses/')

        if self._debug_summaries:
          common.generate_tensor_summaries('MSE', per_example_loss,
                                           self.train_step_counter)

    return tf_agent.LossInfo(total_loss, ()), tape
