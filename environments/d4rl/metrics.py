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

"""Metrics reflecting true task success for d4rl."""
from typing import Optional, Text

import numpy as np

from tf_agents.metrics import py_metrics
from tf_agents.typing import types
from tf_agents.utils import nest_utils

# See d4rl/hand_manipulation_suite/door_v0.py;l=141. This
# determines whether the rollout was a success.
NUM_SUCCESS_STEPS = 25


class D4RLSuccessMetric(py_metrics.StreamingMetric):
  """Computes the task success for D4RL environments."""

  def __init__(self,
               env,
               name = 'D4RLSuccess',
               buffer_size = 10,
               batch_size = None):
    """Creates an AverageReturnMetric."""
    assert not env.batched
    self._env = env
    super(D4RLSuccessMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._cnt = 0
    self._successes = []
    pass

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    self._cnt += 1
    if self._env.get_info() is None:
      # Last step doesn't return info.
      return
    success = 1.0 if self._env.get_info()['goal_achieved'] else 0.
    self._successes.append(success)
    lasts = trajectory.is_last()
    if np.any(lasts):
      is_last = np.where(lasts)
      rollout_success = (1.0 if np.sum(self._successes) > NUM_SUCCESS_STEPS
                         else 0.)
      rollout_success = np.asarray(rollout_success, np.float32)

      if rollout_success.shape is ():  # pylint: disable=literal-comparison
        rollout_success = nest_utils.batch_nested_array(rollout_success)

      self.add_to_buffer(rollout_success[is_last])
