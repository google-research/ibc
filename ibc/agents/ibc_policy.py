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

"""Modified ActorPolicy that will generate sample actions to evaluate."""
from typing import Optional, Text

import gin
from ibc.ibc.agents import mcmc
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import nest_map
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils

tfd = tfp.distributions


def generate_registration_functions(policy, policy_network, strategy):
  """Generates a tf.function and a concrete function matching policy calls."""
  batched_network_input_spec = tensor_spec.add_outer_dims_nest(
      (policy.time_step_spec.observation, policy.action_spec),
      outer_dims=(None,))
  batched_step_type_spec = tensor_spec.add_outer_dims_nest(
      policy.time_step_spec.step_type, outer_dims=(None,))
  batched_policy_state_spec = tensor_spec.add_outer_dims_nest(
      policy.policy_state_spec, outer_dims=(None,))

  @tf.function
  def _create_variables(specs, training, step_type, network_state):
    return strategy.run(
        policy_network,
        args=(specs,),
        kwargs={
            'step_type': step_type,
            'network_state': network_state,
            'training': training
        })

  # Called for the side effect of tracing the function so that it is captured by
  # the saved model.
  _create_variables.get_concrete_function(
      batched_network_input_spec,
      step_type=batched_step_type_spec,
      network_state=batched_policy_state_spec,
      training=tensor_spec.TensorSpec(shape=(), dtype=tf.bool))

  return _create_variables


@tfp.experimental.register_composite
class MappedCategorical(tfp.distributions.Categorical):
  """Categorical distribution that maps classes to specific values."""

  def __init__(self,
               logits=None,
               probs=None,
               mapped_values=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name='MappedCategorical'):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
        set of Categorical distributions. The first `N - 1` dimensions index
        into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities of a set
        of Categorical distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of probabilities for each class. Only one of `logits` or `probs`
        should be passed in.
      mapped_values: Values that map to each category.
      dtype: The type of the event samples (default: int32).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    self._mapped_values = mapped_values
    super(MappedCategorical, self).__init__(
        logits=logits,
        probs=probs,
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  def mode(self, name='mode'):
    """Mode of the distribution."""
    mode = super(MappedCategorical, self).mode(name)
    return tf.gather(self._mapped_values, [mode], batch_dims=0)

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    """Generate samples of the specified shape."""
    # TODO(oars): Fix for complex sample_shapes
    sample = super(MappedCategorical, self).sample(
        sample_shape=sample_shape, seed=seed, name=name, **kwargs)
    return tf.gather(self._mapped_values, [sample], batch_dims=0)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return tfp.distributions.Categorical._parameter_properties(
        dtype=dtype, num_classes=num_classes)


@gin.configurable
class IbcPolicy(tf_policy.TFPolicy):
  """Class to build Actor Policies."""

  def __init__(self,
               time_step_spec,
               action_spec,
               action_sampling_spec,
               actor_network,
               policy_state_spec = (),
               info_spec = (),
               num_action_samples=2**14,
               clip = True,
               training = False,
               name = None,
               use_dfo = False,
               use_langevin = True,
               inference_langevin_noise_scale = 1.0,
               optimize_again = False,
               again_stepsize_init = 1e-1,
               again_stepsize_final = 1e-5,
               late_fusion=False,
               obs_norm_layer=None,
               act_denorm_layer=None):
    """Builds an Actor Policy given an actor network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      action_sampling_spec: A nest of `BoundedTensorSpec` representing the
        bounded actions as seen in the expert dataset.
      actor_network: An instance of a `tf_agents.networks.network.Network` to be
        used by the policy. The network will be called with `call(observation,
        step_type, policy_state)` and should return `(actions_or_distributions,
        new_state)`.
      policy_state_spec: A nest of TensorSpec representing the policy_state. If
        not set, defaults to actor_network.state_spec.
      info_spec: A nest of `TensorSpec` representing the policy info.
      num_action_samples: Number of samples to evaluate for every element in the
        call batch.
      clip: Whether to clip actions to spec before returning them.
      training: Whether the network should be called in training mode.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
      use_dfo: Whether to use dfo mcmc at inference time.
      use_langevin: Whether to use Langevin mcmc at inference time.
      inference_langevin_noise_scale: Scale for Langevin noise at inference
        time.
      optimize_again: Whether or not to run another round of Langevin
        steps but this time with no noise.
      again_stepsize_init: If optimize_again, Langevin schedule init.
      again_stepsize_final: If optimize_again, Langevin schedule final.
      late_fusion: If True, observation tiling must be done in the
        actor_network to match the action.
      obs_norm_layer: Use to normalize observations.
      act_denorm_layer: Use to denormalize actions for inference.

    Raises:
      ValueError: if `actor_network` is not of type `network.Network`.
    """
    if isinstance(actor_network, network.Network):
      # To work around create_variables we force stuff to be build beforehand.
      # TODO(oars): Generalize networks.create_variables
      assert actor_network.built

      if not policy_state_spec:
        policy_state_spec = actor_network.state_spec

    self._action_sampling_spec = action_sampling_spec

    self._action_sampling_minimum = tf.Variable(
        self._action_sampling_spec.minimum,
        trainable=False,
        name='sampling/minimum')
    self._action_sampling_maximum = tf.Variable(
        self._action_sampling_spec.maximum,
        trainable=False,
        name='sampling/maximum')

    self._num_action_samples = num_action_samples
    self._use_dfo = use_dfo
    self._use_langevin = use_langevin
    self._inference_langevin_noise_scale = inference_langevin_noise_scale
    self._optimize_again = optimize_again
    self._again_stepsize_init = again_stepsize_init
    self._again_stepsize_final = again_stepsize_final
    self._late_fusion = late_fusion
    self._obs_norm_layer = obs_norm_layer
    self._act_denorm_layer = act_denorm_layer

    # TODO(oars): Add validation from the network

    self._actor_network = actor_network
    self._training = training

    super(IbcPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        clip=clip,
        name=name)

  def _variables(self):
    return self._actor_network.variables + [
        self._action_sampling_minimum, self._action_sampling_maximum
    ]

  def _distribution(self, time_step, policy_state):
    # Use first observation to figure out batch/time sizes as they should be the
    # same across all observations.
    observations = time_step.observation
    if isinstance(observations, dict) and 'rgb' in observations:
      observations['rgb'] = tf.image.convert_image_dtype(
          observations['rgb'], dtype=tf.float32)

    if self._obs_norm_layer is not None:
      observations = self._obs_norm_layer(observations)
      if isinstance(self._obs_norm_layer, nest_map.NestMap):
        observations, _ = observations

    single_obs = tf.nest.flatten(observations)[0]
    batch_size = tf.shape(single_obs)[0]

    if self._late_fusion:
      maybe_tiled_obs = observations
    else:
      maybe_tiled_obs = nest_utils.tile_batch(observations,
                                              self._num_action_samples)
    # Initialize.
    # TODO(peteflorence): support other initialization options.
    action_samples = tensor_spec.sample_spec_nest(
        self._action_sampling_spec,
        outer_dims=(batch_size * self._num_action_samples,))

    # MCMC.
    probs = 0
    if self._use_dfo:
      probs, action_samples, _ = mcmc.iterative_dfo(
          self._actor_network,
          batch_size,
          maybe_tiled_obs,
          action_samples,
          policy_state,
          num_action_samples=self._num_action_samples,
          min_actions=self._action_sampling_spec.minimum,
          max_actions=self._action_sampling_spec.maximum,
          training=self._training,
          late_fusion=self._late_fusion,
          tfa_step_type=time_step.step_type)

    if self._use_langevin:
      action_samples = mcmc.langevin_actions_given_obs(
          self._actor_network,
          maybe_tiled_obs,
          action_samples,
          policy_state=policy_state,
          num_action_samples=self._num_action_samples,
          min_actions=self._action_sampling_spec.minimum,
          max_actions=self._action_sampling_spec.maximum,
          training=False,
          tfa_step_type=time_step.step_type,
          noise_scale=1.0)

      # Run a second optimization, a trick for more precise
      # inference.
      if self._optimize_again:
        action_samples = mcmc.langevin_actions_given_obs(
            self._actor_network,
            maybe_tiled_obs,
            action_samples,
            policy_state=policy_state,
            num_action_samples=self._num_action_samples,
            min_actions=self._action_sampling_spec.minimum,
            max_actions=self._action_sampling_spec.maximum,
            training=False,
            tfa_step_type=time_step.step_type,
            sampler_stepsize_init=self._again_stepsize_init,
            sampler_stepsize_final=self._again_stepsize_final,
            noise_scale=self._inference_langevin_noise_scale)

      probs = mcmc.get_probabilities(self._actor_network,
                                     batch_size,
                                     self._num_action_samples,
                                     maybe_tiled_obs,
                                     action_samples,
                                     training=False)

    if self._act_denorm_layer is not None:
      action_samples = self._act_denorm_layer(action_samples)
      if isinstance(self._act_denorm_layer, nest_map.NestMap):
        action_samples, _ = action_samples

    # Make a distribution for sampling.
    distribution = MappedCategorical(
        probs=probs, mapped_values=action_samples)
    return policy_step.PolicyStep(distribution, policy_state)
