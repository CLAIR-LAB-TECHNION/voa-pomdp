import time
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from numpy import pi
import time


def get_robot(ip="192.168.0.10", trails=3):
    for i in range(trails):
        try:
            return urx.Robot(ip, urFirm=5.9, use_rt=False)
        except urx.ursecmon.TimeoutException as e:
            print(f"TimeoutError when trying to get robot, trying again {i+1}/{trails}")
        time.sleep(0.1)


rob = get_robot("192.168.0.10")
# robotiqgrip = Robotiq_Two_Finger_Gripper(rob, socket_host="192.168.0.10", socket_port=50002, socket_name="gripper1")
robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
robotiqgrip.close_gripper()
time.sleep(0.5)
robotiqgrip.open_gripper()
robotiqgrip.close_gripper()


home = [0, -pi/2, 0, -pi/2, 0, 0]

rob.movej(home, acc=0.2, vel=0.2)

goal = home
goal[0] = pi/4
rob.movej(goal, acc=0.2, vel=0.2)

rob.close()
