from copy import deepcopy
import numpy as np
import typer
from matplotlib import pyplot as plt
from experiments_lab.block_stacking_env import LabBlockStackingEnv
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.manipulation.utils import ur5e_2_distribute_blocks_from_block_positions_dists, \
    ur5e_2_collect_blocks_from_positions
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default)
from lab_ur_stack.vision.utils import (lookat_verangle_distance_to_robot_config, detections_plots_no_depth_as_image,
                                       detections_plots_with_depth_as_image)
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from experiments_lab.fixed_policy_sense_until_positive import FixedSenseUntilPositivePolicy

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
    r2_controller.speed, r2_controller.acceleration = 3.0, 3.0

    env = LabBlockStackingEnv(n_blocks, 10, r1_controller, r2_controller, gt, camera, position_estimator)
    policy = FixedSenseUntilPositivePolicy()

    initial_belief = BlocksPositionsBelief(n_blocks, workspace_x_lims_default, workspace_y_lims_default,
                                           initial_positions_mus[:n_blocks], initial_positions_sigmas[:n_blocks])

    env.reset_from_distribution(initial_belief.block_beliefs, perform_cleanup=False)
    secret_block_pos = env._block_positions

    current_belief = deepcopy(initial_belief)
    steps_left = env.max_steps
    prev_observation = (False, steps_left)
    while steps_left > 0 and current_belief.n_blocks_on_table > 0:
        action = policy(current_belief, prev_observation)

        plot_im = plot_all_blocks_beliefs(current_belief,
                                          actual_states=secret_block_pos,
                                          ret_as_image=True, )
        plt.figure(figsize=(5, 4*env.n_blocks), dpi=512)
        plt.imshow(plot_im)
        plt.axis("off")
        plt.title(f"Action: \n {action[0]}, ({action[1]:.4f}, {action[2]:.4f}) \n Steps left: {steps_left}")
        plt.show()

        observation = env.step(*action)
        steps_left = observation[1]

        if action[0] == "sense":
            current_belief.update_from_point_sensing_observation(action[1], action[2], observation[0])
        elif action[0] == "attempt_stack":
            if observation[0]:  # successful pickup
                current_belief.update_from_successful_pick(action[1], action[2])

        prev_observation = observation

    env.reset_from_positions(secret_block_pos)


if __name__ == "__main__":
    app()
