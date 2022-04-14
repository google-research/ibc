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

"""Oracles (experts) for particle tasks."""

import random

from ibc.environments.particle import particle
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class ParticleOracle(py_policy.PyPolicy):
  """Oracle moving between two different goals."""

  def __init__(self,
               env,
               wait_at_first_goal = 1,
               multimodal = False,
               goal_threshold = 0.01):
    """Create oracle.

    Args:
      env: Environment.
      wait_at_first_goal: How long to wait at first goal, once you get there.
                          Encourages memory.
      multimodal: If true, go to one or other goal.
      goal_threshold: How close is considered good enough.
    """
    super(ParticleOracle, self).__init__(env.time_step_spec(),
                                         env.action_spec())
    self._env = env
    self._np_random_state = np.random.RandomState(0)

    assert wait_at_first_goal > 0
    self.wait_at_first_goal = wait_at_first_goal
    assert goal_threshold > 0.
    self.goal_threshold = goal_threshold

    self.multimodal = multimodal
    self.reset()

  def reset(self):
    self.steps_at_first_goal = 0
    self.goal_order = ['pos_first_goal', 'pos_second_goal']
    if self.multimodal:
      # Choose a random goal order.
      random.shuffle(self.goal_order)

  def _action(self, time_step,
              policy_state):

    if time_step.is_first():
      self.reset()

    first_goal_key = self.goal_order[0]
    second_goal_key = self.goal_order[1]
    obs = time_step.observation

    gt_goals = self._env.obs_log[0]
    dist = np.linalg.norm(obs['pos_agent'] - gt_goals[first_goal_key])
    if dist < self.goal_threshold:
      self.steps_at_first_goal += 1

    if self.steps_at_first_goal < self.wait_at_first_goal:
      act = np.copy(gt_goals[first_goal_key])
    else:
      act = np.copy(gt_goals[second_goal_key])

    return policy_step.PolicyStep(action=act)
