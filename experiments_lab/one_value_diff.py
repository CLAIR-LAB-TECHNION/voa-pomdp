from copy import deepcopy
import typer
from matplotlib import pyplot as plt
from experiments_lab.block_stacking_env import LabBlockStackingEnv
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera, RealsenseCameraWithRecording
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from modeling.policies.fixed_policy_sense_until_positive import FixedSenseUntilPositivePolicy
from modeling.policies.pouct_planner_policy import POUCTPolicy
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from experiments_lab.experiment_manager import ExperimentManager
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config
import numpy as np


initial_positions_mus = [[-0.8, -0.75], [-0.6, -0.65]]
initial_positions_sigmas = [[0.04, 0.02], [0.05, 0.07]]

# fixed help config:
lookat = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0]
lookat[1] -=0.2
help_verangle = 50
help_horangle = 55
help_distance = 1.15


app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 2,):
    camera = RealsenseCameraWithRecording()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 2.0, 4.0

    initial_belief = BlocksPositionsBelief(n_blocks, workspace_x_lims_default, workspace_y_lims_default,
                                           initial_positions_mus[:n_blocks], initial_positions_sigmas[:n_blocks])

    env = LabBlockStackingEnv(n_blocks, 5, r1_controller, r2_controller, gt, camera, position_estimator)
    # policy = FixedSenseUntilPositivePolicy()
    policy = POUCTPolicy(initial_belief, env.max_steps, goal_tower_position, stacking_reward=env.stacking_reward,
                         sensing_cost_coeff=env.sensing_cost_coeff, stacking_cost_coeff=env.stacking_cost_coeff,
                         finish_ahead_of_time_reward_coeff=env.finish_ahead_of_time_reward_coeff, max_planning_depth=6,
                         show_progress=True)

    experiment_mgr = ExperimentManager(env=env, policy=policy, )


    experiment_mgr.sample_value_difference_experiments(n_blocks=n_blocks,
                                                       min_prior_std=0.01,
                                                       max_prior_std=0.1,
                                                       dirname="experiments/2blocks/")


if __name__ == "__main__":
    app()
