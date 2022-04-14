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

"""Simple particle environment."""

import collections
import copy
import os
from typing import Union

import gin
import gym
from gym import spaces
from gym.envs import registration
from ibc.environments.particle import particle_metrics
from ibc.environments.particle import particle_viz
import matplotlib.pyplot as plt
import numpy as np


@gin.configurable
class ParticleEnv(gym.Env):
  """Simple particle environment with gym wrapper.

  The env is configurable but the default is:
  "go to the green goal, then the blue goal"

  A key feature of this environment is that it is N-dimensional, i.e. the
  observation space is:
    4N:
      - position of the agent (N dimensions)
      - velocity of the agent (N dimensions)
      - position of the first goal (N dimensions)
      - position of the second goal (N dimensions)
  and action space is:
    N:
      - position setpoint for the agent (N dimensions)

  Also configurable instead to:
  - *wait at the first goal for some time (forces memory)
  - *randomly go to either the green OR blue goal first (multimodal)
  - not observe velocity information (can also force more memory usage,
    since the velocity can be used to infer intent)

  * = set in the Oracle params, not the env.

  Key functions:
  - reset() --> state
  - step(action) --> state, reward, done, info
  """

  def get_metrics(self, num_episodes):
    metrics = [
        particle_metrics.AverageFirstGoalDistance(
            self, buffer_size=num_episodes),
        particle_metrics.AverageSecondGoalDistance(
            self, buffer_size=num_episodes),
        particle_metrics.AverageFinalSecondGoalDistance(
            self, buffer_size=num_episodes),
        particle_metrics.AverageSuccessMetric(
            self, buffer_size=num_episodes)
    ]
    success_metric = metrics[-1]
    return metrics, success_metric

  @gin.configurable
  def __init__(
      self,
      n_steps = 50,
      n_dim = 2,
      hide_velocity = False,
      seed = None,
      dt = 0.005,  # 0.005 = 200 Hz
      repeat_actions = 10,  # 10 makes control 200/10 = 20 Hz
      k_p = 10.,
      k_v = 5.,
      goal_distance = 0.05
  ):
    """Creates an env instance with options, see options below.

    Args:
      n_steps: # of steps until done.
      n_dim: # of dimensions.
      hide_velocity: whether or not to hide velocity info from agent.
      seed: random seed.
      dt: timestep for internal simulation (not same as control rate)
      repeat_actions: repeat the action this many times, each for dt.
      k_p: P gain in PD controller. (p for position)
      k_v: D gain in PD controller. (v for velocity)
      goal_distance: Acceptable distances to goals for success.
    """
    self.reset_counter = 0
    self.img_save_dir = None

    self.n_steps = n_steps
    self.goal_distance = goal_distance

    self.n_dim = n_dim
    self.hide_velocity = hide_velocity
    self._rng = np.random.RandomState(seed=seed)

    self.dt = dt
    self.repeat_actions = repeat_actions
    # Make sure is a multiple.
    assert int(1/self.dt) % self.repeat_actions == 0

    self.k_p = k_p
    self.k_v = k_v
    self.action_space = spaces.Box(low=0., high=1., shape=(self.n_dim,),
                                   dtype=np.float32)
    self.observation_space = self._create_observation_space()
    self.reset()

  def _create_observation_space(self):
    obs_dict = collections.OrderedDict(
        pos_agent=spaces.Box(low=0., high=1., shape=(self.n_dim,),
                             dtype=np.float32),
        # TODO(peteflorence): is this the actual max for vel_agent?
        vel_agent=spaces.Box(low=-1e2, high=1e2, shape=(self.n_dim,),
                             dtype=np.float32),
        pos_first_goal=spaces.Box(low=0., high=1., shape=(self.n_dim,),
                                  dtype=np.float32),
        pos_second_goal=spaces.Box(low=0., high=1., shape=(self.n_dim,),
                                   dtype=np.float32)
    )

    if self.hide_velocity:
      del obs_dict['vel_agent']

    return spaces.Dict(obs_dict)

  def seed(self, seed=None):
    self._rng = np.random.RandomState(seed=seed)

  def reset(self):
    self.reset_counter += 1
    self.steps = 0
    # self.obs_log and self.act_log hold internal state,
    # will be useful for plotting.
    self.obs_log = []
    self.act_log = []
    self.new_actions = []

    obs = dict()
    obs['pos_agent'] = self._rng.rand(self.n_dim).astype(np.float32)
    obs['vel_agent'] = np.zeros((self.n_dim)).astype(np.float32)
    obs['pos_first_goal'] = self._rng.rand(self.n_dim).astype(np.float32)
    obs['pos_second_goal'] = self._rng.rand(self.n_dim).astype(np.float32)

    self.obs_log.append(obs)

    self.min_dist_to_first_goal = np.inf
    self.min_dist_to_second_goal = np.inf
    return self._get_state()

  def _get_state(self):
    return copy.deepcopy(self.obs_log[-1])

  def _internal_step(self, action, new_action):
    if new_action:
      self.new_actions.append(len(self.act_log))
    self.act_log.append({'pos_setpoint': action})
    obs = self.obs_log[-1]
    # u = k_p (x_{desired} - x) + k_v (xdot_{desired} - xdot)
    # xdot_{desired} is always (0, 0) -->
    # u = k_p (x_{desired} - x) - k_v (xdot)
    u_agent = self.k_p * (action - obs['pos_agent']) - self.k_v * (
        obs['vel_agent'])
    new_xy_agent = obs['pos_agent'] + obs['vel_agent'] * self.dt
    new_velocity_agent = obs['vel_agent'] + u_agent * self.dt
    obs = copy.deepcopy(obs)
    obs['pos_agent'] = new_xy_agent
    obs['vel_agent'] = new_velocity_agent
    self.obs_log.append(obs)

  def dist(self, goal):
    current_position = self.obs_log[-1]['pos_agent']
    return np.linalg.norm(current_position - goal)

  def _get_reward(self, done):
    """Reward is 1.0 if agent hits both goals and stays at second."""

    # This also statefully updates these values.
    self.min_dist_to_first_goal = min(
        self.dist(self.obs_log[0]['pos_first_goal']),
        self.min_dist_to_first_goal)
    self.min_dist_to_second_goal = min(
        self.dist(self.obs_log[0]['pos_second_goal']),
        self.min_dist_to_second_goal)

    def _reward(thresh):
      reward_first = True if self.min_dist_to_first_goal < thresh else False
      reward_second = True if self.min_dist_to_second_goal < thresh else False
      return 1.0 if (reward_first and reward_second and done) else 0.0

    reward = _reward(self.goal_distance)
    return reward

  @property
  def succeeded(self):
    thresh = self.goal_distance
    hit_first = True if self.min_dist_to_first_goal < thresh else False
    hit_second = True if self.min_dist_to_second_goal < thresh else False
    # TODO(peteflorence/coreylynch: this doesn't work for multimodal version)
    current_distance_to_second = self.dist(self.obs_log[0]['pos_second_goal'])
    still_at_second = True if current_distance_to_second < thresh else False
    return hit_first and hit_second and still_at_second

  def step(self, action):
    self.steps += 1
    self._internal_step(action, new_action=True)
    for _ in range(self.repeat_actions - 1):
      self._internal_step(action, new_action=False)
    state = self._get_state()
    done = True if self.steps >= self.n_steps else False
    reward = self._get_reward(done)
    return state, reward, done, {}

  def render(self, mode='rgb_array'):
    fig = plt.figure()
    fig.add_subplot(111)
    if self.n_dim == 2:
      fig, _ = particle_viz.visualize_2d(self.obs_log, self.act_log)
    else:
      fig, _ = particle_viz.visualize_nd(self.obs_log, self.act_log)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

  def set_img_save_dir(self, summary_dir):
    self.img_save_dir = os.path.join(summary_dir, 'imgs')
    os.makedirs(self.img_save_dir, exist_ok=True)

  def save_image(self, traj):
    if traj.is_last():
      assert self.img_save_dir is not None
      if self.n_dim == 2:
        fig, _ = particle_viz.visualize_2d(self.obs_log, self.act_log)
        filename = os.path.join(self.img_save_dir,
                                str(self.reset_counter).zfill(6)+'_2d.png')
        fig.savefig(filename)
        plt.close(fig)
      fig, _ = particle_viz.visualize_nd(self.obs_log, self.act_log)
      filename = os.path.join(self.img_save_dir,
                              str(self.reset_counter).zfill(6)+'_nd.png')
      fig.savefig(filename)
      plt.close(fig)


# Make sure we only register once to allow us to reload the module in colab for
# debugging.
if 'Particle-v0' in registration.registry.env_specs:
  del registration.registry.env_specs['Particle-v0']

registration.register(id='Particle-v0', entry_point=ParticleEnv)
