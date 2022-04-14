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

"""Writes out a video."""
import os

from absl import logging
try:
  from ibc.ibc.utils import mp4_video_wrapper  # pylint: disable=g-import-not-at-top
except ImportError:
  from ibc.ibc.utils import oss_mp4_video_wrapper as mp4_video_wrapper  # pylint: disable=g-import-not-at-top
from ibc.ibc.utils import strategy_policy  # pylint: disable=g-import-not-at-top
from tf_agents.drivers import py_driver


def make_video(agent, env, root_dir, step, strategy):
  """Creates a video of a single rollout from the current policy."""
  policy = strategy_policy.StrategyPyTFEagerPolicy(
      agent.policy, strategy=strategy)
  video_path = os.path.join(root_dir, 'videos', 'ttl=7d', 'vid_%d.mp4' % step)
  if not hasattr(env, 'control_frequency'):
    # Use this control freq for d4rl envs, which don't have a control_frequency
    # attr.
    control_frequency = 30
  else:
    control_frequency = env.control_frequency
  video_env = mp4_video_wrapper.Mp4VideoWrapper(
      env, control_frequency, frame_interval=1, video_filepath=video_path)
  driver = py_driver.PyDriver(video_env, policy, observers=[], max_episodes=1)
  time_step = video_env.reset()
  initial_policy_state = policy.get_initial_state(1)
  driver.run(time_step, initial_policy_state)
  video_env.close()  # Closes only the video env, not the underlying env.
  logging.info('Wrote video for step %d to %s', step, video_path)
