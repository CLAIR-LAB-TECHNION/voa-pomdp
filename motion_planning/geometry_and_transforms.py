"""
Transformations between different coordinate systems, where the world coordinate system is
the one in the motion planner (aligned with UR5e_1)
"""

from klampt.math import se3, so3
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation as R
from camera.configurations_and_params import camera_in_ee
from motion_planning.motion_planner import MotionPlanner


class GeometryAndTransforms:
    def __init__(self, motion_planner: MotionPlanner):
        self.motion_planner = motion_planner
        self.robot_name_mapping = motion_planner.robot_name_mapping

    @classmethod
    def from_motion_planner(cls, motion_planner):
        return cls(motion_planner)

    @classmethod
    def build(cls):
        mp = MotionPlanner()
        return cls(mp)

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

    def world_to_robot_ee_transform(self, robot_name, config):
        """
        Returns the transformation from the world coordinate system to the end effector of the robot.
        """
        ree_2_w = self.robot_ee_to_world_transform(robot_name, config)
        return se3.inv(ree_2_w)

    def robot_ee_to_world_transform(self, robot_name, config):
        """
        Returns the transformation from the world coordinate system to the end effector of the robot.
        """
        return self.motion_planner.get_forward_kinematics(robot_name, config)

    def camera_to_ee_transform(self,):
        """
        Returns the transformation from the end effector to the camera.
        """
        # we assume camera is z forward, x right, y down (like in the image). this is already the ee frame orientation,
        # so we just need to translate it
        return se3.from_translation(np.array(camera_in_ee))

    def ee_to_camera_transform(self, ):
        """
        Returns the transformation from the camera to the end effector.
        """
        # we assume camera is z forward, x right, y down (like in the image). this is already the ee frame orientation,
        # so we just need to translate it
        return se3.from_translation(-np.array(camera_in_ee))

    def point_world_to_camera(self, point_world, robot_name, config):
        """
        Transforms a point from the world coordinate system to the camera coordinate system.
        """
        transform_w_to_ee = self.world_to_robot_ee_transform(robot_name, config)
        transform_ee_to_camera = self.ee_to_camera_transform()
        transform_w_to_camera = se3.mul(transform_ee_to_camera, transform_w_to_ee)
        return se3.apply(transform_w_to_camera, point_world)

    def point_camera_to_world(self, point_camera, robot_name, config):
        """
        Transforms a point from the camera coordinate system to the world coordinate system.
        """
        transform_camera_to_ee = self.camera_to_ee_transform()
        transform_ee_to_w = self.robot_ee_to_world_transform(robot_name, config)
        transform_camera_to_w = se3.mul(transform_ee_to_w, transform_camera_to_ee)
        return se3.apply(transform_camera_to_w, point_camera)

    def get_gripper_facing_downwards_6d_pose_robot_frame(self, robot_name, point_world, rz):
        """
        Returns a pose in the robot frame where the gripper is facing downwards.
        """
        point_robot = self.point_world_to_robot(robot_name, point_world)

        rotation_down = R.from_euler('xyz', [np.pi, 0, 0])
        rotation_z = R.from_euler('z', rz)
        combined_rotation = rotation_z * rotation_down
        # after spending half day on it, It turns out that UR works with rotvec :(
        r = combined_rotation.as_rotvec(degrees=False)

        return np.concatenate([point_robot, r])

    def get_tilted_pose_6d_for_sensing(self, robot_name, point_world):
        """
        Returns a pose in the robot frame where the gripper is tilted for sensing
        """
        point_robot = self.point_world_to_robot(robot_name, point_world)

        rotation_euler = R.from_euler('xyz', [1.2 * pi, 0.15*pi, 0])
        r = rotation_euler.as_rotvec(degrees=False)

        return np.concatenate([point_robot, r])

    def rotvec_to_so3(self, rotvec):
        return so3.from_rotation_vector(rotvec)

    def orientation_world_to_robot(self, robot_name, orientation_world):
        """
        Transforms an orientation from the world coordinate system to the robot's coordinate system.
        """
        raise NotImplementedError("Not tested yet! It should, and you can remove this line, but be careful!"
                                  "Turns out need to work with rotvec. fix this!")
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = se3.inv(robot.link(0).getTransform())
        return so3.mul(world_to_robot[0], orientation_world)

    def orientation_robot_to_world(self, robot_name, orientation_robot):
        """
        Transforms an orientation from the robot's coordinate system to the world coordinate system.
        """
        raise NotImplementedError("Not tested yet! It should, and you can remove this line, but be careful!"
                                  "Turns out need to work with rotvec. fix this!")
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = robot.link(0).getTransform()
        return so3.mul(world_to_robot[0], orientation_robot)

