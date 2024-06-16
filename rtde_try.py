from robot_inteface.robot_interface import RobotInterfaceWithGripper


robot = RobotInterfaceWithGripper("192.168.0.10")


# freq = 125
# rtde_c = rtdectrl("192.168.0.10", freq,)
# rtde_r = rtdercv("192.168.0.10", freq)
# rtde_io = rtdeio("192.168.0.10", freq)
# gripper = TwoFG7("192.168.0.10", 0)
# time.sleep(0.5)
#
#
# home = [0, -pi/2, 0, -pi/2, 0, 0]
# rtde_c.moveJ(q=home, speed=0.5, acceleration=0.5)
# gripper.twofg_grip_external(35.0, 40, 25)
#
# goal = home.copy()
# goal[0] = pi/5
# rtde_c.moveJ(q=goal, speed=0.5, acceleration=0.5)
#
# pose = rtde_r.getActualTCPPose()
# target_pose = pose
# target_pose[2] -= 0.1
# rtde_c.moveL(target_pose, speed=0.1, acceleration=0.1)
#
# target_pose[1] -= 0.1
# rtde_c.moveJ_IK(target_pose, speed=0.5, acceleration=0.5)
# gripper.twofg_grip_external(0.0, 40, 25)
#
# rtde_c.moveJ(q=home, speed=0.5, acceleration=0.5)




