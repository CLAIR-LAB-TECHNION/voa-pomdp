import json
import os
from typing import List
import cv2
import numpy as np
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation.manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2
from camera.realsense_camera import RealsenseCamera
import time
from PIL import Image


app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True})
def main():
    camera = RealsenseCamera()
    robot = ManipulationController.build_from_robot_name_and_ip(ur5e_1["ip"], ur5e_1["name"])

    point = [0.45, -0.02]

    robot.plan_and_move_to_xyzrz(*point, 0.2, 0, 0.2, 0.2)

    im_arr, depth = camera.get_frame()
    camera.plot_rgb(im_arr)
    # save image:
    im = Image.fromarray(im_arr)
    im.save("im_for_calibration/image.png")

    print("ee pose: ", robot.getActualTCPPose())
    print("robot config: ", robot.getActualQ())
    print("robot tcp offset: ", robot.getTCPOffset())

    camera_offset = [-0.0075, -0.105, 0.0395 - 0.15]
    camera_above_point = [point[0] - camera_offset[0], point[1] + camera_offset[1]]
    robot.plan_and_move_to_xyzrz(*camera_above_point, 0.2, 0, 0.2, 0.2)

    im_arr, depth = camera.get_frame()
    camera.plot_rgb(im_arr)
    # save image:
    im = Image.fromarray(im_arr)
    im.save("im_for_calibration/image_with_pred_tcp_offset.png")


if __name__ == "__main__":
    app()

