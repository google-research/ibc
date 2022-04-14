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

"""Metrics for the blocks environment."""
from typing import Optional, Text

import numpy as np

from tf_agents.metrics import py_metrics
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage


class AverageSuccessMetric(py_metrics.StreamingMetric):
  """Computes the average success of the environment."""

  def __init__(self,
               env,
               name = 'AverageSuccessMetric',
               buffer_size = 10,
               batch_size = None):
    """Creates an AverageReturnMetric."""
    self._np_state = numpy_storage.NumpyState()
    self._env = env
    # Set a dummy value on self._np_state so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.success = np.float64(0)
    super(AverageSuccessMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.success = np.zeros(shape=(batch_size,), dtype=np.float64)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    success = self._np_state.success

    is_first = np.where(trajectory.is_first())
    success[is_first] = 0

    success += self._env.succeeded

    is_last = np.where(trajectory.is_last())
    self.add_to_buffer(success[is_last])


class AverageFinalGoalDistance(py_metrics.StreamingMetric):
  """Computes the average success of the environment."""

  def __init__(self,
               env,
               name = 'AverageFinalGoalDistance',
               buffer_size = 10,
               batch_size = None):
    """Creates an AverageReturnMetric."""
    self._env = env
    super(AverageFinalGoalDistance, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    pass

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    lasts = trajectory.is_last()
    if np.any(lasts):
      is_last = np.where(lasts)
      goal_distance = np.asarray(self._env.goal_distance, np.float32)

      if goal_distance.shape is ():  # pylint: disable=literal-comparison
        goal_distance = nest_utils.batch_nested_array(goal_distance)

      self.add_to_buffer(goal_distance[is_last])
