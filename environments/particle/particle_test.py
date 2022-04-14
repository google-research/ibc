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

"""Tests for Particle Env."""

from ibc.environments.particle import particle
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import utils
from tf_agents.utils import test_utils


class ParticleEnvTest(test_utils.TestCase):

  def testEnv(self):
    env = particle.ParticleEnv()
    env = suite_gym.wrap_env(env)
    self.assertIsInstance(env, py_environment.PyEnvironment)
    utils.validate_py_environment(env)


if __name__ == '__main__':
  test_utils.main()
