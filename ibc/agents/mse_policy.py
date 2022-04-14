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

"""Light wrapper on actor_policy for MSE agents."""

import tensorflow as tf
from tf_agents.networks import nest_map
from tf_agents.policies import actor_policy


class MsePolicyWrapper(actor_policy.ActorPolicy):
  """Just applies normalization and typecasts for the MSE/MDN networks."""

  def __init__(self, time_step_spec, action_spec, cloning_network,
               obs_norm_layer):
    super(MsePolicyWrapper, self).__init__(
        time_step_spec, action_spec, cloning_network)
    self._obs_norm_layer = obs_norm_layer

  def _distribution(self, time_step, policy_state):
    observations = time_step.observation
    if isinstance(observations, dict) and 'rgb' in observations:
      observations['rgb'] = tf.image.convert_image_dtype(
          observations['rgb'], dtype=tf.float32)

    if self._obs_norm_layer is not None:
      observations = self._obs_norm_layer(observations)
      if isinstance(self._obs_norm_layer, nest_map.NestMap):
        observations, _ = observations

    time_step = time_step._replace(observation=observations)
    return super(MsePolicyWrapper,
                 self)._distribution(time_step, policy_state)
