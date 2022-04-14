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

"""Implicit BC agent."""

import copy
import functools
from typing import Optional, Text

import gin
from ibc.ibc.agents import base_agent
from ibc.ibc.agents import ibc_policy
from ibc.ibc.agents import mcmc
from ibc.ibc.losses import ebm_loss
from ibc.ibc.losses import gradient_loss
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class ImplicitBCAgent(base_agent.BehavioralCloningAgent):
  """TFAgent, implementing implicit behavioral cloning."""

  def __init__(self,
               time_step_spec,
               action_spec,
               action_sampling_spec,
               cloning_network,
               optimizer,
               obs_norm_layer=None,
               act_norm_layer=None,
               act_denorm_layer=None,
               num_counter_examples=256,
               debug_summaries = False,
               summarize_grads_and_vars = False,
               train_step_counter = None,
               name = None,
               fraction_dfo_samples=0.,
               fraction_langevin_samples=1.0,
               ebm_loss_type='info_nce',
               late_fusion=False,
               compute_mse=False,
               run_full_chain_under_gradient=False,
               return_full_chain=True,
               add_grad_penalty=True,
               grad_norm_type='inf',
               softmax_temperature=1.0):
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
    self._num_counter_examples = num_counter_examples
    self._fraction_dfo_samples = fraction_dfo_samples
    self._fraction_langevin_samples = fraction_langevin_samples
    assert self._fraction_dfo_samples + self._fraction_langevin_samples <= 1.0
    assert self._fraction_dfo_samples >= 0.
    assert self._fraction_langevin_samples >= 0.
    self.ebm_loss_type = ebm_loss_type
    if self.ebm_loss_type == 'cd_kl':
      self.cloning_network_copy = copy.deepcopy(self.cloning_network)
      assert self._run_full_chain_under_gradient

    self._run_full_chain_under_gradient = run_full_chain_under_gradient

    self._return_full_chain = return_full_chain
    self._add_grad_penalty = add_grad_penalty
    self._grad_norm_type = grad_norm_type

    self._softmax_temperature = softmax_temperature

    self._late_fusion = late_fusion
    self._compute_mse = compute_mse
    if self._compute_mse:
      self._mse = tf.keras.losses.MeanSquaredError(
          reduction=tf.keras.losses.Reduction.NONE)

    # Collect policy would normally be used for data collection. In a BCAgent
    # we don't expect to use it, unless we want to upgrade this to a DAGGER like
    # setup.
    collect_policy = ibc_policy.IbcPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        action_sampling_spec=action_sampling_spec,
        actor_network=cloning_network,
        late_fusion=late_fusion,
        obs_norm_layer=self._obs_norm_layer,
        act_denorm_layer=self._act_denorm_layer,
    )
    if self.ebm_loss_type == 'info_nce':
      self._kl = tf.keras.losses.KLDivergence(
          reduction=tf.keras.losses.Reduction.NONE)

    policy = greedy_policy.GreedyPolicy(collect_policy)

    super(ImplicitBCAgent, self).__init__(
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
    # ** Note **: Obs spec includes time dim. but hilighted here since we have
    # to deal with it.
    # Observation: [B x T x obs_spec]
    # Action:      [B x act_spec]
    observations, actions = experience

    # Use first observation to figure out batch/time sizes as they should be the
    # same across all observations.
    single_obs = tf.nest.flatten(observations)[0]
    batch_size = tf.shape(single_obs)[0]

    # Now tile and setup observations to be: [B * n+1 x obs_spec]
    # TODO(peteflorence): could potentially save memory by not calling
    # tile_batch both here and in _make_counter_example_actions...
    # but for a later optimization.
    # They only differ by having one more tiled obs or not.
    if self._late_fusion:
      maybe_tiled_obs = observations
    else:
      maybe_tiled_obs = nest_utils.tile_batch(observations,
                                              self._num_counter_examples + 1)

    # [B x 1 x act_spec]
    expanded_actions = tf.nest.map_structure(
        functools.partial(tf.expand_dims, axis=1), actions)

    # generate counter examples outside gradient tape
    if not self._run_full_chain_under_gradient:
      counter_example_actions, combined_true_counter_actions, chain_data = (
          self._make_counter_example_actions(observations,
                                             expanded_actions, batch_size,
                                             training))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      if variables_to_train:
        tape.watch(variables_to_train)
      with tf.name_scope('loss'):

        # generate counter examples inside gradient tape
        if self._run_full_chain_under_gradient:
          counter_example_actions, combined_true_counter_actions, chain_data = (
              self._make_counter_example_actions(observations,
                                                 expanded_actions, batch_size,
                                                 training))

        if self._compute_mse:
          actions_size_n = tf.broadcast_to(
              expanded_actions, (batch_size, self._num_counter_examples,
                                 expanded_actions.shape[-1]))
          mse_counter_examples = self._mse(counter_example_actions,
                                           actions_size_n)
          mse_counter_examples = common.aggregate_losses(
              per_example_loss=mse_counter_examples).total_loss

        if self._late_fusion:
          # Do one cheap forward pass.
          obs_embeddings = self.cloning_network.encode(maybe_tiled_obs,
                                                       training=True)
          # Tile embeddings to match actions.
          obs_embeddings = nest_utils.tile_batch(
              obs_embeddings, self._num_counter_examples + 1)
          # Feed in the embeddings as "prior embeddings"
          # (with no subsequent step). Network does nothing with unused_obs.
          unused_obs = maybe_tiled_obs
          network_inputs = (unused_obs,
                            tf.stop_gradient(combined_true_counter_actions))
          # [B * n+1]
          predictions, _ = self.cloning_network(
              network_inputs, training=training,
              observation_encoding=obs_embeddings)

        else:
          network_inputs = (maybe_tiled_obs,
                            tf.stop_gradient(combined_true_counter_actions))
          # [B * n+1]
          predictions, _ = self.cloning_network(
              network_inputs, training=training)
        # [B, n+1]
        predictions = tf.reshape(predictions,
                                 [batch_size, self._num_counter_examples + 1])

        if self.ebm_loss_type == 'cd_kl':
          grad_flow_network_inputs = (maybe_tiled_obs,
                                      combined_true_counter_actions)
        else:
          grad_flow_network_inputs = None
        per_example_loss, debug_dict = self._compute_ebm_loss(
            batch_size, predictions, counter_example_actions, expanded_actions,
            grad_flow_network_inputs)

        if self._add_grad_penalty:
          grad_loss = gradient_loss.grad_penalty(
              self.cloning_network,
              self._grad_norm_type,
              batch_size,
              chain_data,
              maybe_tiled_obs,
              combined_true_counter_actions,
              training,
          )
          per_example_loss += grad_loss
        else:
          grad_loss = None

        # TODO(peteflorence): add energy regularization?

        # Aggregate losses uses some TF magic to make sure aggregation across
        # TPU replicas does the right thing. It does mean we have to calculate
        # per_example_losses though.
        agg_loss = common.aggregate_losses(
            per_example_loss=per_example_loss,
            sample_weight=weights,
            regularization_loss=self.cloning_network.losses)
        total_loss = agg_loss.total_loss

        losses_dict = {
            'ebm_total_loss': total_loss
        }

        losses_dict.update(debug_dict)
        if grad_loss is not None:
          losses_dict['grad_loss'] = tf.reduce_mean(grad_loss)
        if self._compute_mse:
          losses_dict['mse_counter_examples'] = tf.reduce_mean(
              mse_counter_examples)

        opt_dict = dict()
        if chain_data is not None and chain_data.energies is not None:
          energies = chain_data.energies
          opt_dict['overall_energies_avg'] = tf.reduce_mean(energies)
          first_energies = energies[0]
          opt_dict['first_energies_avg'] = tf.reduce_mean(first_energies)
          final_energies = energies[-1]
          opt_dict['final_energies_avg'] = tf.reduce_mean(final_energies)

        if chain_data is not None and chain_data.grad_norms is not None:
          grad_norms = chain_data.grad_norms
          opt_dict['overall_grad_norms_avg'] = tf.reduce_mean(grad_norms)
          first_grad_norms = grad_norms[0]
          opt_dict['first_grad_norms_avg'] = tf.reduce_mean(first_grad_norms)
          final_grad_norms = grad_norms[-1]
          opt_dict['final_grad_norms_avg'] = tf.reduce_mean(final_grad_norms)

        losses_dict.update(opt_dict)

        # I want to put this under a different name scope.
        # But I'm not sure why tf agents won't log it.
        # common.summarize_scalar_dict(
        #     optimization_dict, step=self.train_step_counter,
        #     name_scope='Optimize/')

        common.summarize_scalar_dict(
            losses_dict, step=self.train_step_counter, name_scope='Losses/')

        if self._debug_summaries:
          common.generate_tensor_summaries('ebm_loss', per_example_loss,
                                           self.train_step_counter)

        # This is a bit of a hack, but it makes it so can compute eval loss,
        # including with various metrics.
        if training:
          return tf_agent.LossInfo(total_loss, ()), tape
        else:
          return losses_dict

  def _compute_ebm_loss(
      self,
      batch_size,  # B
      predictions,  # [B x n+1] with true in column [:, -1]
      counter_example_actions,  # [B x n x act_spec]
      expanded_actions,  # [B x 1 x act_spec]
      grad_flow_network_inputs):
    if self.ebm_loss_type == 'info_nce':
      per_example_loss, debug_dict = ebm_loss.info_nce(
          predictions, batch_size, self._num_counter_examples,
          self._softmax_temperature, self._kl)
    elif self.ebm_loss_type == 'cd':
      per_example_loss, debug_dict = ebm_loss.cd(predictions)
    elif self.ebm_loss_type == 'cd_kl':
      predictions_copy, _ = self.cloning_network_copy(
          grad_flow_network_inputs, training=False)
      predictions_copy = tf.reshape(
          predictions_copy, [batch_size, self._num_counter_examples + 1])
      per_example_loss, debug_dict = ebm_loss.cd_kl(predictions,
                                                    counter_example_actions,
                                                    predictions_copy)
    elif self.ebm_loss_type == 'clipped_cd':
      actions_size_n = tf.broadcast_to(
          expanded_actions,
          (batch_size, self._num_counter_examples, expanded_actions.shape[-1]))
      per_example_loss, debug_dict = ebm_loss.clipped_cd(
          predictions, counter_example_actions, actions_size_n, soft=False)
    elif self.ebm_loss_type == 'soft_clipped_cd':
      actions_size_n = tf.broadcast_to(
          expanded_actions,
          (batch_size, self._num_counter_examples, expanded_actions.shape[-1]))
      per_example_loss, debug_dict = ebm_loss.clipped_cd(
          predictions, counter_example_actions, actions_size_n, soft=True)
    else:
      raise ValueError('Unsupported EBM loss type.')

    return per_example_loss, debug_dict

  def _make_counter_example_actions(
      self,
      observations,  # B x obs_spec
      expanded_actions,  # B x 1 x act_spec
      batch_size,
      training):
    """Given observations and true actions, create counter example actions."""
    # Note that T (time dimension) would be included in obs_spec.

    # Counter example actions [B x num_counter_examples x act_spec]
    random_uniform_example_actions = tensor_spec.sample_spec_nest(
        self._action_sampling_spec,
        outer_dims=(batch_size, self._num_counter_examples))

    # If not optimizing, just return.
    if (self._fraction_dfo_samples == 0.0 and
        self._fraction_langevin_samples == 0.0):
      counter_example_actions = random_uniform_example_actions
      chain_data = None
    else:
      # Reshape to put B and num counter examples on same tensor dimenison
      # [B*num_counter_examples x act_spec]
      random_uniform_example_actions = tf.reshape(
          random_uniform_example_actions,
          (batch_size * self._num_counter_examples, -1))

      if self._late_fusion:
        maybe_tiled_obs_n = observations
      else:
        maybe_tiled_obs_n = nest_utils.tile_batch(observations,
                                                  self._num_counter_examples)

      dfo_opt_counter_example_actions = None
      if self._fraction_dfo_samples > 0.:
        if self._return_full_chain:
          raise NotImplementedError('Not implemented to return dfo chain.')

        # Use all uniform actions to seed the optimization,
        # even though we will only pick a subset later.
        _, dfo_opt_counter_example_actions, _ = mcmc.iterative_dfo(
            self.cloning_network,
            batch_size,
            maybe_tiled_obs_n,
            random_uniform_example_actions,
            policy_state=(),
            num_action_samples=self._num_counter_examples,
            min_actions=self._action_sampling_spec.minimum,
            max_actions=self._action_sampling_spec.maximum,
            training=False,
            late_fusion=self._late_fusion,
            tfa_step_type=())

      lang_opt_counter_example_actions = None
      if self._fraction_langevin_samples > 0.:
        # TODO(peteflorence): in the case of using a fraction <1.0,
        # we could reduce the amount in langevin that are optimized.
        langevin_return = mcmc.langevin_actions_given_obs(
            self.cloning_network,
            maybe_tiled_obs_n,
            random_uniform_example_actions,
            policy_state=(),
            min_actions=self._action_sampling_spec.minimum,
            max_actions=self._action_sampling_spec.maximum,
            training=False,
            tfa_step_type=(),
            return_chain=self._return_full_chain,
            grad_norm_type=self._grad_norm_type,
            num_action_samples=self._num_counter_examples)
        if self._return_full_chain:
          lang_opt_counter_example_actions, chain_data = langevin_return
        else:
          lang_opt_counter_example_actions = langevin_return
          chain_data = None

      list_of_counter_examples = []

      fraction_init_samples = (1. - self._fraction_dfo_samples -
                               self._fraction_langevin_samples)

      # Compute indices based on fractions.
      init_num_indices = int(fraction_init_samples * self._num_counter_examples)
      dfo_num_indices = (
          int(self._fraction_dfo_samples * self._num_counter_examples))
      langevin_num_indices = (
          int(self._fraction_langevin_samples * self._num_counter_examples))
      residual = (
          self._num_counter_examples - init_num_indices - dfo_num_indices -
          langevin_num_indices)
      assert residual >= 0
      # If there was a rounding that caused a residual, ascribe those to init.
      init_num_indices += residual

      used_index = 0
      if init_num_indices > 0:
        some_init_counter_example_actions = tf.reshape(
            random_uniform_example_actions,
            (batch_size, self._num_counter_examples,
             -1))[:, :init_num_indices, :]
        used_index += init_num_indices
        list_of_counter_examples.append(some_init_counter_example_actions)

      if dfo_num_indices > 0.:
        next_index = used_index + dfo_num_indices
        some_dfo_counter_example_actions = tf.reshape(
            dfo_opt_counter_example_actions,
            (batch_size, self._num_counter_examples,
             -1))[:, used_index:next_index, :]
        used_index = next_index
        list_of_counter_examples.append(some_dfo_counter_example_actions)

      if langevin_num_indices > 0.:
        next_index = used_index + langevin_num_indices
        some_lang_counter_example_actions = tf.reshape(
            lang_opt_counter_example_actions,
            (batch_size, self._num_counter_examples,
             -1))[:, used_index:next_index, :]
        used_index = next_index
        list_of_counter_examples.append(some_lang_counter_example_actions)

      assert used_index == self._num_counter_examples

      counter_example_actions = tf.concat(list_of_counter_examples, axis=1)
      counter_example_actions = tf.reshape(
          counter_example_actions, (batch_size, self._num_counter_examples, -1))

    def concat_and_squash_actions(counter_example, action):
      return tf.reshape(
          tf.concat([counter_example, action], axis=1),
          [-1] + self._action_spec.shape.as_list())

    # Batch consists of num_counter_example rows followed by 1 true action.
    # [B * (n + 1) x act_spec]
    combined_true_counter_actions = tf.nest.map_structure(
        concat_and_squash_actions, counter_example_actions, expanded_actions)

    return counter_example_actions, combined_true_counter_actions, chain_data
