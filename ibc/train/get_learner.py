# coding=utf-8
# Copyright 2024 The Reach ML Authors.
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

"""Define the learner."""
import os
from absl import logging
from ibc.ibc.agents import ibc_policy
from tf_agents.train import learner
from tf_agents.train import triggers


def get_learner(loss_type,
                root_dir,
                agent,
                train_step,
                train_data_fn,
                fused_train_steps,
                strategy,
                checkpoint_interval=5000):
  """Defines BC learner."""
  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weights.
  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)

  extra_concrete_functions = []
  # TODO(peteflorence, oars): fix serialization when Policy/Network don't have
  # matching tensor specs, due to float-casting of images.
  if loss_type == 'ebm':
    try:
      cloning_network_fn = ibc_policy.generate_registration_functions(
          agent.policy, agent.cloning_network, strategy)
      extra_concrete_functions = [('cloning_network', cloning_network_fn)]
    except ValueError:
      logging.warning('Unable to generate concrete functions. Skipping.')
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir,
      agent,
      train_step,
      interval=1000,
      extra_concrete_functions=extra_concrete_functions,
      use_nest_path_signatures=False,
      save_greedy_policy=loss_type != 'mdn')

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=100)
  ]

  def dataset_fn():
    train_data, _ = train_data_fn()
    return train_data

  bc_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      dataset_fn,
      triggers=learning_triggers,
      checkpoint_interval=checkpoint_interval,
      summary_interval=fused_train_steps,
      strategy=strategy,
      run_optimizer_variable_init=False)
  return bc_learner
