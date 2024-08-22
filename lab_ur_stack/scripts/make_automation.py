from copy import deepcopy
import typer
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from experiments_lab.block_stacking_env import LabBlockStackingEnv
from lab_ur_stack.manipulation.utils import ur5e_2_collect_blocks_from_positions
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)

from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_no_depth_as_image, detections_plots_with_depth_as_image
import numpy as np


initial_positions_mus = [[-0.8, -0.75], [-0.6, -0.65]]
initial_positions_sigmas = [[0.04, 0.02], [0.05, 0.07]]

# fixed help config:
lookat = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0]
lookat[1] +=0.05
# vertical_angle = 35
# horizontal_angle = 20
# distance = 1
vertical_angle = 25
horizontal_angle = 0
distance = 0.7


app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 4,
         use_depth: bool = 1, ):
    camera = RealsenseCamera()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 2, 1.2

    env = LabBlockStackingEnv(n_blocks, 5, r1_controller, r2_controller, gt, camera, position_estimator)
    help_config = lookat_verangle_horangle_distance_to_robot_config(lookat, vertical_angle, horizontal_angle,
                                                                    distance, gt, "ur5e_1")


    env.r1_controller.plan_and_move_home(speed=2, acceleration=1)
    env.r2_controller.plan_and_move_home(speed=2, acceleration=1)

    positions = sensing(env, use_depth, help_config)

    # r2 collect blocks
    manipulation(env, positions)

    test = True
    if test:
        predicted_n_blocks = len(positions)
        if predicted_n_blocks == n_blocks:
            print("Test passed")
    test_num = 1
    while len(positions) > 0:
        if test_num == 1:
            help_config = change_help_config(env, 2)
            test_num = 2
        positions = sensing(env, use_depth, help_config)
        pos_num = manipulation(env, positions)
        if pos_num == 0:
            break


def sensing(env, use_depth, help_config):
    env.r1_controller.plan_and_moveJ(help_config)
    im, depth = env.camera.get_frame_rgb()
    if use_depth:
        positions, annotations = env.position_estimator.get_block_positions_depth(im, depth, help_config)
        plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                       workspace_x_lims_default, workspace_y_lims_default)
    else:
        positions, annotations = env.position_estimator.get_block_position_plane_projection(im, help_config, plane_z=-0.02)
        plot_im = detections_plots_no_depth_as_image(annotations[0], annotations[1], positions,
                                                     workspace_x_lims_default, workspace_y_lims_default)
    # plot in hires:
    plt.figure(figsize=(12, 12), dpi=512)
    plt.imshow(plot_im)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)
    env.r1_controller.plan_and_move_home(speed=2, acceleration=1.2)
    return positions


def manipulation(env, positions):
    if len(positions) == 0:
        print("No blocks detected")
        return len(positions)
    ur5e_2_collect_blocks_from_positions(positions, env.r2_controller)
    return None


def change_help_config(env, pos_num=1):
    global vertical_angle, horizontal_angle, distance, lookat
    if pos_num == 1:
        vertical_angle = 40
        horizontal_angle = 27
        distance = 0.5
        help_config = lookat_verangle_horangle_distance_to_robot_config(lookat, vertical_angle, horizontal_angle, distance,
                                                                    env.gt, "ur5e_1")
    elif pos_num == 2:
        vertical_angle = 25
        horizontal_angle = 0
        distance = 0.7
        help_config = lookat_verangle_horangle_distance_to_robot_config(lookat, vertical_angle, horizontal_angle,
                                                                        distance,
                                                                        env.gt, "ur5e_1")
    return help_config


if __name__ == "__main__":
    app()
