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

"""Defines the tf agent."""

import enum

from ibc.ibc.agents import ibc_agent
from ibc.ibc.agents import mdn_agent
from ibc.ibc.agents import mse_agent
import tensorflow as tf


class LossType(enum.Enum):
  EBM = 'ebm'
  MSE = 'mse'
  MDN = 'mdn'


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Implements learning rate warmup."""

  def __init__(self, lr, d_model=32, warmup_steps=4000):
    super(WarmupSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    self.lr = lr

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * self.lr


def get_agent(loss_type,
              time_step_tensor_spec,
              action_tensor_spec,
              action_sampling_spec,
              obs_norm_layer,
              act_norm_layer,
              act_denorm_layer,
              learning_rate,
              use_warmup,
              cloning_network,
              train_step,
              decay_steps):
  """Creates tfagent."""
  if use_warmup:
    learning_rate_schedule = WarmupSchedule(lr=learning_rate)
  else:
    learning_rate_schedule = (
        tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps=decay_steps, decay_rate=0.99))
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

  if loss_type == LossType.EBM.value:
    agent_class = ibc_agent.ImplicitBCAgent
  elif loss_type == LossType.MSE.value:
    agent_class = mse_agent.MseBehavioralCloningAgent
  elif loss_type == LossType.MDN.value:
    agent_class = mdn_agent.MdnBehavioralCloningAgent
  else:
    raise ValueError("Unsupported loss type, can't retrieve an agent.")

  agent = agent_class(
      time_step_spec=time_step_tensor_spec,
      action_spec=action_tensor_spec,
      action_sampling_spec=action_sampling_spec,
      obs_norm_layer=obs_norm_layer,
      act_norm_layer=act_norm_layer,
      act_denorm_layer=act_denorm_layer,
      cloning_network=cloning_network,
      optimizer=optimizer,
      train_step_counter=train_step)
  agent.initialize()
  return agent
