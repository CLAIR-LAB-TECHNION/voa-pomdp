from rtde_control import RTDEControlInterface as rtdectrl
from rtde_receive import RTDEReceiveInterface as rtdercv
from rtde_io import RTDEIOInterface as rtdeio
from numpy import pi
import time
import urx


freq = 125
rtde_c = rtdectrl("192.168.0.10", freq,)
rtde_r = rtdercv("192.168.0.10", freq)
rtde_io = rtdeio("192.168.0.10", freq)
time.sleep(0.5)




home = [0, -pi/2, 0, -pi/2, 0, 0]
rtde_c.moveJ(q=home, speed=0.5, acceleration=0.5)

goal = home.copy()
goal[0] = pi/5
rtde_c.moveJ(q=goal, speed=0.5, acceleration=0.5)

pose = rtde_r.getActualTCPPose()
target_pose = pose
target_pose[2] -= 0.1
rtde_c.moveL(target_pose, speed=0.1, acceleration=0.1)

target_pose[1] -= 0.1
rtde_c.moveJ_IK(target_pose, speed=0.5, acceleration=0.5)

rtde_c.moveJ(q=home, speed=0.5, acceleration=0.5)




