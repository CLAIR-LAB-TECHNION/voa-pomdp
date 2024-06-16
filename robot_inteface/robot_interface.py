from rtde_control import RTDEControlInterface as rtdectrl
from rtde_receive import RTDEReceiveInterface as rtdercv
from rtde_io import RTDEIOInterface as rtdeio
from twofg7_gripper import TwoFG7
import numpy as np
from numpy import pi


home_config = [0, -pi/2, 0, -pi/2, 0, 0]


class RobotInterface(rtdectrl, rtdeio, rtdercv):
    def __init__(self, robot_ip, freq=125):
        rtdectrl.__init__(self, robot_ip, freq)
        rtdercv.__init__(self, robot_ip, freq)
        rtdeio.__init__(self, robot_ip, freq)

    def move_home(self, speed=0.2, acceleration=0.2):
        self.moveJ(q=home_config, speed=0.2, acceleration=0.2)


class RobotInterfaceWithGripper(RobotInterface):
    def __init__(self, robot_ip, freq=125, gripper_ip=0):
        super().__init__(robot_ip, freq)
        self.gripper = TwoFG7(robot_ip, gripper_ip)

    def set_gripper(self, width, force, speed):
        self.gripper.twofg_grip_external(width, force, speed)

    def grasp(self):
        min_width = self.gripper.twofg_get_min_external_width()
        self.gripper.twofg_grip_external(min_width, 40, 100)

    def release_grasp(self):
        max_width = self.gripper.twofg_get_max_external_width()
        self.gripper.twofg_grip_external(max_width, 40, 100)
