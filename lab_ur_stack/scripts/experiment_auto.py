#Shiran's code
import typer
import logging
import os
import cv2
import numpy as np
from pyrealsense2 import sensor

from experiments_lab.block_stacking_env import LabBlockStackingEnv
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.manipulation.utils import to_canonical_config
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import goal_tower_position,workspace_x_lims_default, workspace_y_lims_default
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.vision.utils import detections_plots_with_depth_as_image

#Configure logging
logging.basicConfig(level=logging.INFO)

def clean_up_workspace(env):

    logging.info("Cleaning up workspace")

    cleared_blocks_position = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default)]

    # first, clean the tower:
    tower_height_blocks = env.sense_tower_height()
    logging.info(f"tower height is {tower_height_blocks}")
    for i in range(tower_height_blocks):
        start_height = 0.04 * (tower_height_blocks - i) + 0.12
        env.r2_controller.pick_up(goal_tower_position[0],
                                       goal_tower_position[1],
                                       0,
                                       start_height)
        env.r2_controller.plan_and_move_to_xyzrz(cleared_blocks_position[0],
                                                      cleared_blocks_position[1],
                                                      z=0.15,
                                                      rz=0)
        env.r2_controller.release_grasp()


    env.r2_controller.plan_and_move_to_xyzrz(cleared_blocks_position[0], goal_tower_position[1], z=0.15, rz=0)

    #setup the camera configuration for optimal block detection
    lookat = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0]
    sensor_config = env.lookat_verangle_horangle_distance_to_robot_config(lookat, 60, 35, 0.7, gt=env.gt, robot_name="ur5e_1")
    sensor_config = to_canonical_config(sensor_config)
    env.r1_controller.move_to(sensor_config)

    #cepture the image and detect the blocks
    im, depth_im = env.camera.capture_image_and_depth()
    positions, annotations = env.position_estimator.get_block_positions(im, depth_im, sensor_config, max_dedection = 10)
    plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions, workspace_x_lims_default, workspace_y_lims_default, depth_im)

    for position in positions:
        env.r2_controller.pick_up(position[0], position[1], 0, start_height = 0.15)
        env.r1_controller.plan_and_move_to_xyzrz(cleared_blocks_position[0], cleared_blocks_position[1], z=0.15, rz=0)
        env.r1_controller.release_grasp()

    logging.info("Cleaning up workspace done")

    return cv2.cvtColor(plot_im, cv2.COLOR_RGB2BGR)

app = typer.Typer()

@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 4,
         use_depth: bool = 1,):

    camera = RealsenseCamera()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)

    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 2, 1.2

    env = LabBlockStackingEnv(n_blocks, 5, r1_controller, r2_controller, gt, camera, position_estimator)
    clean_up_workspace(env)

if __name__ == "__main__":
    app()



