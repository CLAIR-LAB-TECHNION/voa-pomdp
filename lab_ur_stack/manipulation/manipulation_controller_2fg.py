import numpy as np
from lab_ur_stack.manipulation.robot_with_motion_planning import RobotInterfaceWithMP
from lab_ur_stack.robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.robot_inteface.twofg7_gripper import TwoFG7
from lab_ur_stack.robot_inteface.vgc10_gripper import VG10C
from lab_ur_stack.utils import logging_util
import time
import logging
import chime


class ManipulationController2FG(RobotInterfaceWithMP):
    def __init__(self, robot_ip, robot_name, motion_palnner: MotionPlanner,
                 geomtry_and_transofms: GeometryAndTransforms, freq=50, gripper_id=0):
        super().__init__(robot_ip, robot_name, motion_palnner, geomtry_and_transofms, freq)

        self.gripper = TwoFG7(robot_ip, gripper_id)

        self.min_width = self.gripper.twofg_get_min_external_width()
        self.max_width = self.gripper.twofg_get_max_external_width()

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geomtry_and_transofms = GeometryAndTransforms(motion_planner)
        return cls(robot_ip, robot_name, motion_planner, geomtry_and_transofms)

    #### Gripper functions ####
    def set_gripper(self, width, force, speed, wait_time=0.5):
        logging.debug(f"Setting gripper ({self._ip}), width: {width}, force: {force}, speed: {speed}")
        res = self.gripper.twofg_grip_external(width, force, speed)
        if res != 0:
            logging.warning(f"Failed to set gripper ({self._ip}), width: {width}, force: {force}, speed: {speed}")
        time.sleep(wait_time)

    def grasp(self, wait_time=0.5):
        logging.debug(f"Grasping ({self._ip}), min_width: {self.min_width}")
        res = self.gripper.twofg_grip_external(self.min_width, 20, 100)
        if res != 0:
            logging.warning(f"Failed to grasp ({self._ip})")
        time.sleep(wait_time)

    def release_grasp(self, wait_time=0.5):
        logging.debug(f"Releasing grasp ({self._ip}), max_width: {self.max_width}")
        res = self.gripper.twofg_grip_external(self.max_width, 20, 100)
        if res != 0:
            logging.warning(f"Failed to release grasp ({self._ip})")
        time.sleep(wait_time)

    def is_object_gripped(self):
        return self.gripper.twofg_get_grip_detected()

    #### Pick and place functions ####

    def pick_up(self, x, y, rz, start_height=0.2, replan_from_home_if_failed=True):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} picking up at {x}{y}{rz} with start height {start_height}")

        # move above pickup location:
        res = self.plan_and_move_to_xyzrz(x, y, start_height, rz)

        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_move_to_xyzrz(x, y, start_height, rz)
            if not res:
                chime.error()
                return

        above_pickup_config = self.getActualQ()
        self.release_grasp()

        # move down until contact, here we move a little bit slower than drop and sense
        # because the gripper rubber may damage from the object at contact:
        logging.debug(f"{self.robot_name} moving down until contact")
        lin_speed = min(self.linear_speed / 2, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])

        # retract one more centimeter to avoid gripper scratching the surface:
        self.moveL_relative([0, 0, 0.01],
                            speed=0.1,
                            acceleration=0.1)
        logging.debug(f"{self.robot_name} grasping and picking up")
        # close gripper:
        self.grasp()
        # move up:
        self.moveJ(above_pickup_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()


    def put_down(self, x, y, rz, start_height=0.2, replan_from_home_if_failed=True):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} putting down at {x}{y}{rz} with start height {start_height}")
        # move above dropping location:
        res = self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
            if not res:
                chime.error()
                return

        above_drop_config = self.getActualQ()

        logging.debug(f"{self.robot_name} moving down until contact to put down")
        # move down until contact:
        lin_speed = min(self.linear_speed, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # release grasp:
        self.release_grasp()
        # back up 10 cm in a straight line :
        self.moveL_relative([0, 0, 0.1], speed=self.linear_speed, acceleration=self.linear_acceleration)
        # move to above dropping location:
        self.moveJ(above_drop_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

    # def sense_height(self, x, y, start_height=0.2):
    #     """
    #     TODO
    #     :param x:
    #     :param y:
    #     :param start_height:
    #     :return:
    #     """
    #     logging.info(f"{self.robot_name} sensing height not tilted! at {x}{y} with start height {start_height}")
    #     self.grasp()
    #
    #     # move above sensing location:
    #     self.plan_and_move_to_xyzrz(x, y, start_height, 0, speed=self.speed, acceleration=self.acceleration)
    #     above_sensing_config = self.getActualQ()
    #
    #     lin_speed = min(self.linear_speed, 0.1)
    #     # move down until contact:
    #     self.moveUntilContact(xd=[0, 0, lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    #     # measure height:
    #     height = self.getActualTCPPose()[2]
    #     # move up:
    #     self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)
    #
    #     # update the motion planner with the new configuration:
    #     self.update_mp_with_current_config()
    #
    #     return height

    def sense_height_tilted(self, x, y, start_height=0.15, replan_from_home_if_failed=True):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} sensing height tilted at {x}{y} with start height {start_height}")
        self.grasp()

        # set end effector to be the tip of the finger
        self.setTcp([0.02, 0.012, 0.15, 0, 0, 0])

        logging.debug(f"moving above sensing point with TCP set to tip of the finger")

        # move above point with the tip tilted:
        pose = self.gt.get_tilted_pose_6d_for_sensing(self.robot_name, [x, y, start_height])
        goal_config = self.find_ik_solution(pose, max_tries=50)
        res = self.plan_and_moveJ(goal_config)

        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return -1

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_moveJ(goal_config)
            if not res:
                chime.error()
                return -1

        above_sensing_config = self.getActualQ()

        logging.debug(f"moving down until contact with TCP set to tip of the finger")

        # move down until contact:
        lin_speed = min(self.linear_speed, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # measure height:
        pose = self.getActualTCPPose()
        height = pose[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        # set back tcp:
        self.setTcp([0, 0, 0.150, 0, 0, 0])
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

        logging.debug(f"height measured: {height}, TCP pose at contact: {pose}")

        return height