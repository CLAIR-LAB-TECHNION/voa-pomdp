from robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
import numpy
from numpy import pi
import time
from motion_planning.motion_planner import MotionPlanner


target_position_world_rob1 = [-0.3, -0.5, 0.25]
target_orientation_world_rob1 = [0, pi, 0]
target_pose_world_rob1 = target_position_world_rob1 + target_orientation_world_rob1
target_position_world_rob2 = [-0.4, -0.7, 0.25]
target_orientation_world_rob2 = [0, pi, 0]
target_pose_world_rob2 = target_position_world_rob2 + target_orientation_world_rob2


robot1 = RobotInterfaceWithGripper("192.168.0.10", 50)
robot2 = RobotInterfaceWithGripper("192.168.0.11", 50)

motion_planner = MotionPlanner()
motion_planner.visualize()

time.sleep(0.2)
robot1.move_home(speed=0.5, acceleration=0.5, asynchronous=True)
robot2.move_home(speed=0.5, acceleration=0.5, asynchronous=False)
time.sleep(0.5)

init_config = robot2.getActualQ()

motion_planner.show_point_vis(target_position_world_rob2)
target_pose_robot = motion_planner.transform_world_to_robot("ur5e_2", target_pose_world_rob2)

target_config = robot2.getInverseKinematics(target_pose_robot)
print("target_config: ", target_config)

motion_planner.ur5e_2.setConfig(motion_planner.config6d_to_klampt(target_config))

path = motion_planner.plan_from_start_to_goal_config("ur5e_2", init_config, target_config)
motion_planner.show_path_vis("ur5e_2", path)

vel, acc, blend = 0.5, 1., 0.01
path = [[*target_config, vel, acc, blend] for target_config in path]
robot2.moveJ(path)

######################

init_config = robot1.getActualQ()

motion_planner.show_point_vis(target_position_world_rob1)
target_pose_robot = motion_planner.transform_world_to_robot("ur5e_1", target_pose_world_rob1)
target_config = robot1.getInverseKinematics(target_pose_robot)
print("target_config: ", target_config)
motion_planner.ur5e_1.setConfig(motion_planner.config6d_to_klampt(target_config))

path = motion_planner.plan_from_start_to_goal_config("ur5e_1", init_config, target_config)
motion_planner.show_path_vis("ur5e_1", path)

vel, acc, blend = 0.5, 1., 0.01
path = [[*target_config, vel, acc, blend] for target_config in path]
robot1.moveJ(path)



