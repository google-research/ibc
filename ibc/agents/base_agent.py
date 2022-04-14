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

"""Base BC agent."""

from typing import Optional

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.typing import types
from tf_agents.utils import eager_utils


class BehavioralCloningAgent(tf_agent.TFAgent):
  """TFAgent, implementing base behavioral cloning agent."""

  def _loss(self,
            experience=None,
            variables_to_train=None,
            weights=None,
            training=False):
    raise NotImplementedError("Implement in subclass.")

  def _train(self,
             experience,
             weights = None):
    variables_to_train = self.cloning_network.trainable_weights
    assert list(variables_to_train), "No variables in the agent's network."
    non_trainable_weights = self.cloning_network.non_trainable_weights

    loss_info, tape = self._loss(
        experience, variables_to_train, weights=weights, training=True)

    tf.debugging.check_numerics(loss_info.loss, "Loss is inf or nan")

    grads = tape.gradient(loss_info.loss, variables_to_train)
    grads_and_vars = list(zip(grads, variables_to_train))

    if self._summarize_grads_and_vars:
      grads_and_vars_with_non_trainable = (
          grads_and_vars + [(None, v) for v in non_trainable_weights])
      eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)

    return loss_info

  def get_eval_loss(self, experience):
    loss_dict = self._loss(experience, training=False)
    return loss_dict
