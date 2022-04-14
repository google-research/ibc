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

"""Defines random action sampling spec."""
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


def get_sampling_spec(action_tensor_spec,
                      min_actions,
                      max_actions,
                      uniform_boundary_buffer,
                      act_norm_layer):
  """Defines action sampling based on min/max action +- buffer.

  Args:
    action_tensor_spec: Action spec.
    min_actions: Per-dimension minimum action values seen in subset
      of training data.
    max_actions: Per-dimension minimum action values seen in subset
      of training data.
    uniform_boundary_buffer: Float, percentage of extra "room" to add to
      minimum/maximum boundary when sampling uniform actions.
    act_norm_layer: Normalizer, needed so can sample over normalized actions.
  Returns:
    sampling_spec: Spec used for sampling random uniform negative actions.
  """
  def generate_boundary_buffered_limits(spec, min_action, max_action):
    # Optionally add a small buffer of extra acting range.
    action_range = max_action - min_action
    min_action -= action_range * uniform_boundary_buffer
    max_action += action_range * uniform_boundary_buffer

    # Clip this range to the envs' min/max.
    # There's no point in sampling outside of the envs' min/max.
    min_action = tf.maximum(spec.minimum, min_action)
    max_action = tf.minimum(spec.maximum, max_action)

    return min_action, max_action

  action_limit_nest = tf.nest.map_structure(
      generate_boundary_buffered_limits, action_tensor_spec, min_actions,
      max_actions)

  # Map up to the spec to avoid iterating over the tuples.
  buffered_min_actions = nest_utils.map_structure_up_to(action_tensor_spec,
                                                        lambda a: a[0],
                                                        action_limit_nest)
  buffered_max_actions = nest_utils.map_structure_up_to(action_tensor_spec,
                                                        lambda a: a[1],
                                                        action_limit_nest)

  normalized_min_actions = act_norm_layer(buffered_min_actions)[0]
  normalized_max_actions = act_norm_layer(buffered_max_actions)[0]

  def bounded_like(spec, min_action, max_action):
    return tensor_spec.BoundedTensorSpec(
        spec.shape,
        spec.dtype,
        minimum=min_action,
        maximum=max_action,
        name=spec.name)

  action_sampling_spec = tf.nest.map_structure(bounded_like, action_tensor_spec,
                                               normalized_min_actions,
                                               normalized_max_actions)
  return action_sampling_spec
