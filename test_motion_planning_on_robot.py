from robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
import numpy
from numpy import pi
import time
from motion_planning.motion_planner import MotionPlanner


robot1 = RobotInterfaceWithGripper("192.168.0.11", 125)
motion_planner = MotionPlanner()
motion_planner.visualize()

time.sleep(0.2)
robot1.move_home(speed=0.5, acceleration=1.0)
init_config = robot1.getActualQ()

target_position_world = [-0.5, -0.8, 0.3]
motion_planner.show_point_vis(target_position_world)
target_orientation_world = [0, pi, 0]
target_pose_world = target_position_world + target_orientation_world

target_pose_robot = motion_planner.transform_world_to_robot("ur5e_2", target_pose_world)

target_config = robot1.getInverseKinematics(target_pose_robot)
print("target_config: ", target_config)

motion_planner.ur5e_2.setConfig(motion_planner.config6d_to_klampt(target_config))

path = motion_planner.plan_from_start_to_goal_config("ur5e_2", init_config, target_config)
motion_planner.show_path_vis("ur5e_2", path)

vel, acc, blend = 0.5, 1., 0.01
path = [[*target_config, vel, acc, blend] for target_config in path]
robot1.moveJ(path)

