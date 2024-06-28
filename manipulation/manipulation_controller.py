import numpy as np
from robot_inteface.robot_interface import RobotInterfaceWithGripper
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from utils import logging_util
import time


class ManipulationController(RobotInterfaceWithGripper):
    """
    Extension for the RobotInterfaceWithGripper to higher level manipulation actions and motion planning.
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
                 geomtry_and_transofms: GeometryAndTransforms, freq=50, gripper_id=0):
        super().__init__(robot_ip, freq, gripper_id)

        logging_util.setup_logging()

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

    def update_mp_with_current_config(self):
        self.motion_planner.update_robot_config(self.robot_name, self.getActualQ())

    def find_ik_solution(self, pose, max_tries=10):
        # try to find the one that is closest to the current configuration:
        solution = self.getInverseKinematics(pose)

        trial = 1
        while self.motion_planner.is_config_feasible(self.robot_name, solution) is False and trial < max_tries:
            print("trial", trial, "failed")
            trial += 1
            # try to find another solution, starting from other random configurations:
            qnear = np.random.uniform(-np.pi, np.pi, 6)
            solution = self.getInverseKinematics(pose)

        return solution

    def plan_and_moveJ(self, q, speed=None, acceleration=None, visualise=True):
        """
        Plan and move to a joint configuration.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        start_config = self.getActualQ()

        if visualise:
            self.motion_planner.vis_config(self.robot_name, q, vis_name="goal_config",
                                           rgba=(0, 1, 0, 0.5))
            self.motion_planner.vis_config(self.robot_name, start_config,
                                           vis_name="start_config", rgba=(1, 0, 0, 0.5))

        # plan until the ratio between length and distance is lower than 3, but stop if 4 seconds have passed
        path = self.motion_planner.plan_from_start_to_goal_config(self.robot_name,
                                                                  start_config,
                                                                  q,
                                                                  max_time=4,
                                                                  max_length_to_distance_ratio=2)
        if visualise:
            self.motion_planner.vis_path(self.robot_name, path)

        self.move_path(path, speed, acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

    def plan_and_move_to_xyzrz(self, x, y, z, rz, speed=None, acceleration=None, visualise=True):
        """
        Plan and move to a position in the world coordinate system, with gripper
        facing downwards rotated by rz.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        target_pose_robot = self.gt.get_gripper_facing_downwards_6d_pose_robot_frame(self.robot_name,
                                                                                     [x, y, z],
                                                                                     rz)

        goal_config = self.find_ik_solution(target_pose_robot, max_tries=50)
        self.plan_and_moveJ(goal_config, speed, acceleration, visualise)
        # motion planner is automatically updated after movement

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
        self.plan_and_move_to_xyzrz(x, y, start_height, rz)
        above_pickup_config = self.getActualQ()

        # move down until contact, here we move a little bit slower than drop and sense
        # because the gripper rubber may damage from the object at contact:
        lin_speed = min(self.linear_speed/2, 0.05)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])

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
        self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        above_drop_config = self.getActualQ()

        # move down until contact:
        self.moveUntilContact(xd=[0, 0, -self.linear_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
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
        self.grasp()

        # move above sensing location:
        self.plan_and_move_to_xyzrz(x, y, start_height, 0, speed=self.speed, acceleration=self.acceleration)
        above_sensing_config = self.getActualQ()

        # move down until contact:
        self.moveUntilContact(xd=[0, 0, -self.linear_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # measure height:
        height = self.getActualTCPPose()[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        return height

    def sense_height_tilted(self, x, y, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        self.grasp()

        # set end effector to be the tip of the finger
        self.setTcp([0.020, 0.012, 0.160, 0, 0, 0])

        # move above point with the tip tilted:
        pose = self.gt.get_tilted_pose_6d_for_sensing(self.robot_name, [x, y, start_height])
        goal_config = self.find_ik_solution(pose, max_tries=50)
        self.plan_and_moveJ(goal_config)

        above_sensing_config = self.getActualQ()

        # move down until contact:
        self.moveUntilContact(xd=[0, 0, -self.linear_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # measure height:
        height = self.getActualTCPPose()[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        # set back tcp:
        self.setTcp([0, 0, 0.150, 0, 0, 0])

        return height



