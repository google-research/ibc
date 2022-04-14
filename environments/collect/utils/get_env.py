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

"""Loads an environment for data collection."""
from absl import logging
import gin
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from tf_agents.environments import suite_gym


@gin.configurable
def get_env(task,
            use_image_obs=True,
            fixed_start_poses=False,
            noisy_ee_pose=False,
            shared_memory_pybullet=False,
            max_episode_steps=None):
  """Loads an environment given the task."""
  del use_image_obs
  del fixed_start_poses
  del noisy_ee_pose
  env_name = block_pushing.build_env_name(
      task,
      shared_memory_pybullet,
      use_image_obs=False,
      use_normalized_env=False)
  logging.info('Loading environment %s (env_name=%s)', task, env_name)
  return suite_gym.load(env_name, max_episode_steps=max_episode_steps)
