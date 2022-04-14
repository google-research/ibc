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

"""Tests for ibc.environments.utils_pybullet."""

import os
import tempfile

from ibc.environments.block_pushing import block_pushing
from ibc.environments.utils.utils_pybullet import read_pybullet_state
from ibc.environments.utils.utils_pybullet import write_pybullet_state
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.environments import suite_gym


class PybulletStateTest(tf.test.TestCase):

  def test_serialize_deserialize(self):
    for env_name in [block_pushing.build_env_name('PUSH', False, False)]:
      env = suite_gym.load(env_name)
      state = [env.get_pybullet_state()]
      task = 'test'

      # Serialize the state to file.
      filename = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), env_name + '.json.zip')
      actions = np.random.rand(1, 2).tolist()
      write_pybullet_state(filename, state, task, actions=actions)
      self.assertTrue(tf.io.gfile.exists(filename))
      data = read_pybullet_state(filename)

      self.assertEqual(data['task'], task)
      self.assertEqual(data['pybullet_state'], state)
      self.assertEqual(data['actions'], actions)

      # Set the state largely for code coverage.
      env.set_pybullet_state(data['pybullet_state'][0])


if __name__ == '__main__':
  tf.test.main()
