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

"""Tests for ibc.environments.xarm_sim_robot."""
import math

from ibc.environments.utils import xarm_sim_robot
from ibc.environments.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import tensorflow.compat.v2 as tf
import pybullet
import pybullet_utils.bullet_client as bullet_client


class XArmSimRobotTest(tf.test.TestCase):

  def setUp(self):
    super(XArmSimRobotTest, self).setUp()

    # To debug we can use the SHARED_MEMORY connection.
    # pybullet.connect(pybullet.SHARED_MEMORY)
    connection_mode = pybullet.SHARED_MEMORY
    connection_mode = pybullet.DIRECT
    self._pybullet_client = bullet_client.BulletClient(connection_mode)
    self._pybullet_client.resetSimulation()
    self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    self._pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)

  def test_arm_loads(self):
    xarm_sim_robot.XArmSimRobot(self._pybullet_client)

  def test_arm_loads_suction(self):
    xarm_sim_robot.XArmSimRobot(self._pybullet_client, end_effector='suction')

  def test_forward_kinematics(self):
    robot = xarm_sim_robot.XArmSimRobot(self._pybullet_client)

    # Pointing down X Axis
    robot.reset_joints([0, math.pi / 2, math.pi, 0, 0, 0])
    x, y, _ = robot.forward_kinematics().translation

    self.assertAlmostEqual(0.714479, x, places=3)
    self.assertAlmostEqual(-0.0006, y, places=3)

    # Pointing down Y Axis
    robot.reset_joints([math.pi / 2, math.pi / 2, math.pi, 0, 0, 0])
    x, y, _ = robot.forward_kinematics().translation

    self.assertAlmostEqual(0.0006, x, places=3)
    self.assertAlmostEqual(0.714479, y, places=3)

  def test_inverse_kinematics(self):
    robot = xarm_sim_robot.XArmSimRobot(self._pybullet_client)
    initial_pose = robot.forward_kinematics()

    rotation = transform.Rotation.from_rotvec([0, math.pi / 2, 0])
    translation = np.array([0.5, 0.0, 0.10])
    target_pose = Pose3d(rotation=rotation, translation=translation)

    robot.reset_joints(robot.inverse_kinematics(target_pose))
    pose = robot.forward_kinematics()

    self.assertFalse(np.all(initial_pose.vec7 == pose.vec7))
    np.testing.assert_almost_equal(pose.vec7, target_pose.vec7, decimal=2)


if __name__ == '__main__':
  tf.test.main()
