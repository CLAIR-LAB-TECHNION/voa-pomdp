import numpy as np
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation.manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2
from camera.realsense_camera import RealsenseCamera
from vision.image_block_position_estimator import ImageBlockPositionEstimator
from manipulation.utils import ur5e_2_distribute_blocks_in_workspace, ur5e_2_collect_blocks_from_positions
from utils.workspace_utils import (workspace_x_lims_default,
                                   workspace_y_lims_default)
from vision.utils import lookat_verangle_distance_to_robot_config


# camera pose:
lookat = (np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0)  # middle of the workspace
verangle = 45
distance = 0.6

app = typer.Typer()

@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 3,):
    camera = RealsenseCamera()

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 1.5, 2.0

    # r2 distribute blocks and clear out
    actual_block_positions = ur5e_2_distribute_blocks_in_workspace(n_blocks, r2_controller)
    r2_controller.move_home(speed=0.5, acceleration=0.5)

    # take image and clear out
    r1_sensing_config = lookat_verangle_distance_to_robot_config(lookat, verangle, distance, gt, ur5e_1["name"])
    r1_controller.plan_and_move_to_config(r1_sensing_config)

    bgr, depth = camera.get_frame_bgr()
    if bgr is None or depth is None:
        raise ValueError("Camera not working")

