"""
Transformations between different coordinate systems, where the world coordinate system is
the one in the motion planner (aligned with UR5e_1)
"""

from klampt.math import se3,so3
from klampt import RobotModel
import numpy as np
from numpy import pi


class GeometryAndTransforms:
    def __init__(self, robot_name_mapping):
        self.robot_name_mapping = robot_name_mapping

    @classmethod
    def from_motion_planner(cls, motion_planner):
        return cls(motion_planner.robot_name_mapping)

    def point_world_to_robot(self, robot_name, point_world):
        """
        Transforms a point from the world coordinate system to the robot's coordinate system.
        """
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = se3.inv(robot.link(0).getTransform())
        return se3.apply(world_to_robot, point_world)

    def point_robot_to_world(self, robot_name, point_robot):
        """
        Transforms a point from the robot's coordinate system to the world coordinate system.
        """
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = robot.link(0).getTransform()
        return se3.apply(world_to_robot, point_robot)

    def get_gripper_facing_downwards_6d_pose_robot_frame(self, robot_name, point_world, rz):
        """
        Returns a pose in the robot frame where the gripper is facing downwards.
        """
        point_robot = self.point_world_to_robot(robot_name, point_world)
        return np.concatenate([point_robot, [0, pi, rz]])

    def orientation_world_to_robot(self, robot_name, orientation_world):
        """
        Transforms an orientation from the world coordinate system to the robot's coordinate system.
        """
        raise NotImplementedError("Not tested yet! It should, and you can remove this line, but be careful!")
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = se3.inv(robot.link(0).getTransform())
        return so3.mul(world_to_robot[0], orientation_world)

    def orientation_robot_to_world(self, robot_name, orientation_robot):
        """
        Transforms an orientation from the robot's coordinate system to the world coordinate system.
        """
        raise NotImplementedError("Not tested yet! It should, and you can remove this line, but be careful!")
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = robot.link(0).getTransform()
        return so3.mul(world_to_robot[0], orientation_robot)

