from copy import deepcopy

import numpy as np
import typer
from matplotlib import pyplot as plt
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.manipulation.utils import ur5e_2_distribute_blocks_from_block_positions_dists,\
    ur5e_2_collect_blocks_from_positions
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default)
from lab_ur_stack.vision.utils import (lookat_verangle_distance_to_robot_config, detections_plots_no_depth_as_image,
                                       detections_plots_with_depth_as_image)
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.belief.belief_plotting import plot_all_blocks_beliefs


initial_positions_mus = [[-0.8, -0.75], [-0.6, -0.65]]
initial_positions_sigmas = [[0.04, 0.02], [0.05, 0.07]]


app = typer.Typer()

@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 2,
         use_depth_for_help: bool = 1, ):
    camera = RealsenseCamera()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 1.0, 1.0

    sensing_config = None  # TODO
    # TODO also visualize it and so on...

    r1_controller.plan_and_move_home(speed=1., acceleration=1.)

    assert len(initial_positions_mus) >= n_blocks, "Not enough initial positions for the blocks"

    initial_belief = BlocksPositionsBelief(n_blocks, workspace_x_lims_default, workspace_y_lims_default,
                                           initial_positions_mus[:n_blocks], initial_positions_sigmas[:n_blocks])
    actual_positions = ur5e_2_distribute_blocks_from_block_positions_dists(initial_belief.block_beliefs, r2_controller)
    plot_all_blocks_beliefs(initial_belief, actual_states=actual_positions)

    # try to pick block 1, sample points from it's position distribution to sense until a positive point is sensed:
    belief = deepcopy(initial_belief)
    negative_points = []
    positive_points = []
    is_positive = False
    while not is_positive:
        block_belief = belief.block_beliefs[0]
        point = block_belief.sample()[0]
        sensed_height = r2_controller.sense_height_tilted(point[0], point[1])
        if sensed_height > 0.03:  # usually sensing is positive at 0.32
            print(f"Positive point sensed at {point}, height: {sensed_height}")
            is_positive = True
            positive_points.append(point)
        else:
            print(f"Negative point sensed at {point}, height: {sensed_height}")
            negative_points.append(point)
        belief.update_from_point_sensing_observation(point[0], point[1], is_positive)
        plot_all_blocks_beliefs(belief, actual_states=actual_positions, negative_sensing_points=negative_points,
                                positive_sensing_points=positive_points)

    # find maximum likelihood within the belief, sample point within the new window:










if __name__ == "__main__":
    app()
