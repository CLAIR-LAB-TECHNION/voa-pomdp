from rtde_control import RTDEControlInterface as rtdectrl
from rtde_receive import RTDEReceiveInterface as rtdercv
from rtde_io import RTDEIOInterface as rtdeio
from robot_inteface.twofg7_gripper import TwoFG7
from numpy import pi
import time


home_config = [0, -pi/2, 0, -pi/2, 0, 0]


# class RobotInterface(rtdectrl, rtdeio, rtdercv):
class RobotInterface(rtdectrl, rtdercv):
    def __init__(self, robot_ip, freq=50):
        rtdectrl.__init__(self, robot_ip, freq)
        rtdercv.__init__(self, robot_ip, freq)
        # rtdeio.__init__(self, robot_ip, freq)

    def move_home(self, speed=0.2, acceleration=0.2, asynchronous=False):
        self.moveJ(q=home_config, speed=speed, acceleration=acceleration, asynchronous=asynchronous)

    def move_path(self, path, speed=0.5, acceleration=0.5, blend_radius=0.05, asynchronous=False):
        path_with_params = [[*target_config, speed, acceleration, blend_radius] for target_config in path]
        # last section should have blend radius 0 otherwise the robot will not reach the last target
        path_with_params[-1][-1] = 0
        self.moveJ(path_with_params, asynchronous=asynchronous)

    def moveL_relative(self, relative_position, speed=0.5, acceleration=0.5, asynchronous=False):
        target_pose = self.getActualTCPPose()
        for i in range(3):
            target_pose[i] += relative_position[i]
        self.moveL(target_pose, speed, acceleration, asynchronous)

class RobotInterfaceWithGripper(RobotInterface):
    def __init__(self, robot_ip, freq=50, gripper_id=0):
        super().__init__(robot_ip, freq)
        self.gripper = TwoFG7(robot_ip, gripper_id)

    def set_gripper(self, width, force, speed):
        self.gripper.twofg_grip_external(width, force, speed)
        time.sleep(1)

    def grasp(self, wait_time=0.5):
        min_width = self.gripper.twofg_get_min_external_width()
        self.gripper.twofg_grip_external(min_width, 20, 100)
        time.sleep(wait_time)

    def release_grasp(self, wait_time=0.5):
        max_width = self.gripper.twofg_get_max_external_width()
        self.gripper.twofg_grip_external(max_width, 20, 100)
        time.sleep(wait_time)
