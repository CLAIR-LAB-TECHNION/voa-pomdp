import numpy as np

from robot_inteface.robot_interface import RobotInterfaceWithGripper
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
import time


class ManipulationController(RobotInterfaceWithGripper):
    """
    Extension for the RobotInterfaceWithGripper to higher level manipulation actions
    """
    # those are angular in radians:
    speed = 1.0
    acceleration = 1.0

    # and this is linear, ratio that makes sense:
    @property
    def linear_speed(self):
        return self.speed * 0.1
    @property
    def linear_acceleration(self):
        return self.acceleration * 0.1

    def __init__(self, robot_ip, robot_name, motion_palnner: MotionPlanner,
                 geomtry_and_transofms: GeometryAndTransforms, freq=125, gripper_id=0):
        super().__init__(robot_ip, freq, gripper_id)
        self.robot_name = robot_name
        self.motion_planner = motion_palnner
        self.gt = geomtry_and_transofms

        motion_palnner.visualize()
        time.sleep(0.2)

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geomtry_and_transofms = GeometryAndTransforms(motion_planner.robot_name_mapping)
        return cls(robot_ip, robot_name, motion_planner, geomtry_and_transofms)

    def plan_and_move_to(self, x, y, z, rz, speed=1.0, acceleration=1.0, visualise=True):
        """
        Plan and move to a position in the world coordinate system, with gripper
        facing downwards rotated by rz.
        """
        target_pose_robot = self.gt.get_gripper_facing_downwards_6d_pose_robot_frame(self.robot_name, [x, y, z], rz)

        goal_config = self.getInverseKinematics(target_pose_robot)
        start_config = self.getActualQ()

        if visualise:
            self.motion_planner.vis_config(self.robot_name, goal_config)

        # plan until the ratio between length and distance is lower than 3, but stop if 4 seconds have passed
        path = self.motion_planner.plan_from_start_to_goal_config(self.robot_name,
                                                                  start_config,
                                                                  goal_config,
                                                                  max_time=4,
                                                                  max_length_to_distance_ratio=2)
        if visualise:
            self.motion_planner.vis_path(self.robot_name, path)

        self.move_path(path, speed, acceleration)
        # print("error from target:", np.array(self.getActualTCPPose() - target_pose_robot))


    def pick_up(self, x, y, rz, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        self.release_grasp()

        # move above pickup location:
        self.plan_and_move_to(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        above_pickup_config = self.getActualQ()

        # move down until contact, here we move a little bit slower than drop and sense
        # because the gripper rubber may damage from the object at contact:
        lin_speed = min(self.linear_speed/2, 0.05)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0],)

        # retract one more centimeter to avoid gripper scratching the surface:
        self.moveL_relative([0, 0, 0.01],
                            speed=0.1,
                            acceleration=0.1)
        # close gripper:
        self.grasp()
        # move up:
        self.moveJ(above_pickup_config, speed=self.speed, acceleration=self.acceleration)

        # TODO measure weight and return if successful or not

    def put_down(self, x, y, rz, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        # move above dropping location:
        self.plan_and_move_to(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        above_drop_config = self.getActualQ()

        # move down until contact:
        self.moveUntilContact(xd=[0, 0, -self.linear_speed, 0, 0, 0],)
        # release grasp:
        self.release_grasp()
        # back up 10 cm in a straight line :
        self.moveL_relative([0, 0, 0.1], speed=self.linear_speed, acceleration=self.linear_acceleration)
        # move to above dropping location:
        self.moveJ(above_drop_config, speed=self.speed, acceleration=self.acceleration)

    def sense_height(self, x, y, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param start_height:
        :return:
        """
        # move above sensing location:
        self.plan_and_move_to(x, y, start_height, 0, speed=self.speed, acceleration=self.acceleration)
        above_sensing_config = self.getActualQ()

        # move down until contact:
        self.moveUntilContact(speed=[0, 0, -self.linear_speed, 0, 0, 0],)
        # measure height:
        height = self.getActualTCPPose()[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        return height