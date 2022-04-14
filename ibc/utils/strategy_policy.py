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

"""A PyTfEagerPolicy that runs under a tf.distribute.Strategy."""

from typing import Optional

from absl import logging
import tensorflow as tf
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class StrategyPyTFEagerPolicy(py_tf_eager_policy.PyTFEagerPolicyBase):
  """Exposes a numpy API for TF policies in Eager mode, using a Strategy."""

  def __init__(self,
               policy,
               strategy,
               batch_time_steps=True):
    time_step_spec = tensor_spec.to_nest_array_spec(policy.time_step_spec)
    action_spec = tensor_spec.to_nest_array_spec(policy.action_spec)
    policy_state_spec = tensor_spec.to_nest_array_spec(policy.policy_state_spec)
    info_spec = tensor_spec.to_nest_array_spec(policy.info_spec)
    self._strategy = strategy
    use_tf_function = True
    super(StrategyPyTFEagerPolicy,
          self).__init__(policy, time_step_spec, action_spec, policy_state_spec,
                         info_spec, use_tf_function, batch_time_steps)

  def _action(self, time_step, policy_state, seed = None):
    if seed is not None and self._use_tf_function:
      logging.warning(
          'Using `seed` may force a retrace for each call to `action`.')
    if self._batch_time_steps:
      time_step = nest_utils.batch_nested_array(time_step)
    # Avoid passing numpy arrays to avoid retracing of the tf.function.
    time_step = tf.nest.map_structure(tf.convert_to_tensor, time_step)

    # This currently only takes the first result from the replicated results.
    # If there is more than one device, other results will be ignored.
    if seed is not None:
      strategy_result = self._strategy.run(
          self._policy_action_fn,
          args=(time_step, policy_state),
          kwargs={'seed': seed})
      local_results = self._strategy.experimental_local_results(strategy_result)
      policy_step = local_results[0]
    else:
      strategy_result = self._strategy.run(
          self._policy_action_fn, args=(time_step, policy_state))
      local_results = self._strategy.experimental_local_results(strategy_result)
      policy_step = local_results[0]
    if not self._batch_time_steps:
      return policy_step
    return policy_step._replace(
        action=nest_utils.unbatch_nested_tensors_to_arrays(policy_step.action),
        # We intentionally do not convert the `state` so it is outputted as the
        # underlying policy generated it (i.e. in the form of a Tensor) which is
        # not necessarily compatible with a py-policy. However, we do so since
        # the `state` is fed back to the policy. So if it was converted, it'd be
        # required to convert back to the original form before calling the
        # method `action` of the policy again in the next step. If one wants to
        # store the `state` e.g. in replay buffer, then we suggest placing it
        # into the `info` field.
        info=nest_utils.unbatch_nested_tensors_to_arrays(policy_step.info))
