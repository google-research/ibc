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

"""Utils for masking goal information from sequence models to test memory."""
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.networks import nest_map

NP_EPS = float(np.finfo(np.float32).eps)


class FlattenNormalizeMaskGoal(object):
  """Flattens, normalizes, and masks observation goal info in correct order.

  Usage:
    # Create mask.
    mask = FlattenNormalizeMaskGoal(
      flatten_layer = flatten_obs_layer,
      sequence_length=20,
      see_goal_every=4,
      goal_keys=['target_translation','target_orientation'],
      )
    # Get obs dict.
    obs = {
      'block_orientation': [batch, seqlen, obs_dim],
      'block_translation': [batch, seqlen, obs_dim],
      'effector_target_translation': [batch, seqlen, obs_dim],
      'effector_translation': [batch, seqlen, obs_dim],
      'target_orientation': [batch, seqlen, obs_dim],
      'target_translation: [batch, seqlen, obs_dim],
      }
    # Get flattened, normalized, goal-masked observation.
    obs = mask(obs)
    assert obs.shape == [batch, seqlen, obs_dim].
  """

  def __init__(self,
               sequence_length,
               see_goal_every,
               goal_keys=None,
               normalize=True,
               obs_norm_layer=None):
    # Normalize params.
    self._normalize = normalize
    self._obs_norm_layer = obs_norm_layer
    # Goal mask params.
    self._goal_mask = None
    self._sequence_length = sequence_length
    self._see_goal_every = see_goal_every
    self._goal_keys = goal_keys
    self._mask_goal = goal_keys is not None

  def _create_mask_lazily(self, observation):
    """Creates a [1, sequence_length, concatenated obs_dim] mask.

    This creates a reusable tensor binary mask (using numpy logic because it's
    simpler). This mask is applied to each training / test tensor observation.

    Args:
      observation: dict of named numpy observations coming from environment. We
        use this to infer the shape of each obs_dim.
        Shape [batch, seqlen, obs_dim].
    Returns:
      full_mask: [1, seqlen, full_obs_dim] binary mask, where full_obs_dim is
        the sum of all the obs_dims in the observation data dict. This is
        broadcastable to any batch size.
    """
    # Allow goal information to be viewable at these indices.
    goal_viewable_indices = np.arange(
        0, self._sequence_length, self._see_goal_every)

    # Create reusable masks for all observations, one for when we are hiding
    # goal info, one for when we are exposing goal info.
    mask_with_goal_hidden = []
    mask_with_goal_exposed = []
    for k in sorted(observation):
      v = observation[k]
      # Get the shape of this observation.
      v_dim = v.shape[-1]
      mask_hidden = np.zeros(v_dim) if k in self._goal_keys else np.ones(v_dim)
      mask_exposed = np.ones(v_dim)
      mask_with_goal_hidden.append(mask_hidden)
      mask_with_goal_exposed.append(mask_exposed)
    mask_with_goal_hidden = np.concatenate(mask_with_goal_hidden)
    mask_with_goal_exposed = np.concatenate(mask_with_goal_exposed)

    # Create the full [1, seqlen, full obs dim] mask.
    full_mask = []
    for time_idx in range(self._sequence_length):
      if time_idx in goal_viewable_indices:
        full_mask.append(mask_with_goal_exposed)
      else:
        full_mask.append(mask_with_goal_hidden)
    full_mask = np.array(full_mask)[np.newaxis, :, :]
    full_mask = tf.convert_to_tensor(full_mask, dtype=tf.float32)
    return full_mask

  def __call__(self, obs_dict):
    # Lazily create goal mask if it doesn't exist.
    if self._goal_mask is None and self._mask_goal:
      self._goal_mask = self._create_mask_lazily(obs_dict)

    # Flatten observation.
    obs = tf.concat(nest_map.NestFlatten()(obs_dict), -1)

    # Normalize observation.
    if self._normalize:
      obs = self._obs_norm_layer(obs)

    # Apply goal mask.
    if self._mask_goal:
      obs *= self._goal_mask

    return obs
