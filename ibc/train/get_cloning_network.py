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

"""Defines the cloning network."""
import copy

import ibc.ibc.utils.constants as constants
from ibc.networks import mlp_ebm
from ibc.networks import mlp_mdn
from ibc.networks import mlp_mse
from ibc.networks import pixel_ebm
from ibc.networks import pixel_mdn
from ibc.networks import pixel_mse
import tensorflow as tf
from tf_agents.specs import tensor_spec


def adjust_img_spec_to_float(obs_tensor_spec):
  """If there are images, adjust spec to be float32."""
  float_obs_tensor_spec = obs_tensor_spec
  for img_key in constants.IMG_KEYS:
    if isinstance(obs_tensor_spec, dict) and img_key in obs_tensor_spec:
      img_spec = obs_tensor_spec[img_key]
      float_obs_tensor_spec = copy.deepcopy(obs_tensor_spec)
      float_obs_tensor_spec[img_key] = tensor_spec.BoundedTensorSpec(
          img_spec.shape,
          dtype=tf.float32,
          minimum=img_spec.minimum,
          maximum=1.0)
  return float_obs_tensor_spec


def get_cloning_network(
    name,
    obs_tensor_spec,
    action_tensor_spec,
    obs_norm_layer,
    act_norm_layer,
    sequence_length,
    act_denorm_layer):
  """Chooses a cloning network based on config.

  Args:
    name: Name of the network to build.
    obs_tensor_spec: A nest of `tf.TypeSpec` representing the
        input observations.
    action_tensor_spec: A nest of `tf.TypeSpec` representing the actions.
    obs_norm_layer: Keras layer to normalize observations.
    act_norm_layer: Keras layer to normalize actions.
    sequence_length: Length of the observation sequence.
    act_denorm_layer: Layer mapping zmuv-normalized actions back to original
      spec.

  Returns:
    A cloning network.
  """
  del obs_norm_layer  # placeholder
  del act_norm_layer
  del sequence_length

  obs_tensor_spec = adjust_img_spec_to_float(obs_tensor_spec)

  if name == 'MLPEBM':
    cloning_network = mlp_ebm.MLPEBM(
        obs_spec=(obs_tensor_spec, action_tensor_spec),
        action_spec=tf.TensorSpec([1]))
  elif name == 'MLPMSE':
    cloning_network = mlp_mse.MLPMSE(
        obs_spec=obs_tensor_spec,
        action_spec=action_tensor_spec,
        act_denorm_layer=act_denorm_layer)
  elif name == 'MLPMDN':
    cloning_network = mlp_mdn.MLPMDN(
        obs_spec=obs_tensor_spec,
        action_spec=action_tensor_spec,
        act_denorm_layer=act_denorm_layer)
  elif name == 'ConvMLPMSE':
    cloning_network = pixel_mse.ConvMLPMSE(
        obs_spec=obs_tensor_spec,
        action_spec=action_tensor_spec,
        act_denorm_layer=act_denorm_layer)
  elif name == 'ConvMLPMDN':
    cloning_network = pixel_mdn.ConvMLPMDN(
        obs_spec=obs_tensor_spec,
        action_spec=action_tensor_spec,
        act_denorm_layer=act_denorm_layer)
  elif name == 'PixelEBM':
    cloning_network = pixel_ebm.PixelEBM(
        obs_spec=obs_tensor_spec,
        action_spec=action_tensor_spec)
  else:
    raise ValueError('Unsupported cloning network %s' % name)
  return cloning_network
