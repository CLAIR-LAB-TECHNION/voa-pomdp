import json
import os
from typing import List
import cv2
import numpy as np
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from robot_inteface.robot_interface import RobotInterfaceWithGripper
from robot_inteface.robots_metadata import ur5e_1, ur5e_2
from camera.realsense_camera import RealsenseCamera
from vision.image_block_position_estimator import ImageBlockPositionEstimator
import time


app = typer.Typer()

workspace_x_lims = [-0.9, -0.54]
workspace_y_lims = [-1.0, -0.55]

@app.command(
    context_settings={"ignore_unknown_options": True})
def main(use_depth: bool = 0):

    # TODO actual positions as input or from file

    camera = RealsenseCamera()
    gt = GeometryAndTransforms.build()
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims, workspace_y_lims, gt)

    robot = RobotInterfaceWithGripper(ur5e_1['ip', ])
    robot.freedriveMode(free_axes=[1]*6)

    while True:
        robot_config = robot.getActualQ()
        rgb, depth = camera.get_frame()

        if use_depth:
            positions, annotations = position_estimator.get_block_positions_depth(rgb, depth, robot_config)

        else:
            positions, annotations = position_estimator.get_block_position_plane_projection(rgb, robot_config)
            plot_detections_no_depth(TODO)

if __name__ == "__main__":
    app()