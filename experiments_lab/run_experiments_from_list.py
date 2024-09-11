import os
import time
import logging
import pandas as pd
import typer
import numpy as np
import chime

from experiments_lab.block_stacking_env import LabBlockStackingEnv
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCameraWithRecording
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.pouct_planner_policy import POUCTPolicy
from modeling.policies.hand_made_policy import HandMadePolicy
from experiments_lab.experiment_manager import ExperimentManager

app = typer.Typer()


@app.command(context_settings={"ignore_unknown_options": True})
def run_experiments(
        n_blocks: int = 4,
        max_steps: int = 20,
        policy_type: str = typer.Option("pouct", help="policy type, pouct or hand_made"),
        max_planning_depth: int = typer.Option(5, help=" Only relevant for POUCT policy."
                                                       "Max planning depth for the planner,"
                                                       "consult before changing it, deepening may cause"
                                                       "distribution shift because rollout policy always tries to pick up"),
        planner_n_iterations: int = typer.Option(2000, help="Only relevant for POUCT policy"
                                                            "Number of iterations for the planner"),
        confidence_for_stack: float = typer.Option(0.6, help="Only relevant for HandMade policy."),
        config_file: str = typer.Option("configurations/experiments_4_blocks.csv"),
        two_experiments_at_a_time: bool = typer.Option(False, help="If True, will run two experiments at a time"
                                                                   "assuming theer are two stacks on the pile"),
):
    results_dir = f"experiments/{n_blocks}blocks"
    os.makedirs(results_dir, exist_ok=True)
    chime.theme('pokemon')

    # Setup environment and policy
    camera = RealsenseCameraWithRecording()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 2.0, 4.0

    # Load experiment configurations
    df = pd.read_csv(config_file)
    logging.info(f"Loaded {len(df)} experiments from {config_file}")

    env = LabBlockStackingEnv(n_blocks, max_steps, r1_controller, r2_controller, gt, camera, position_estimator)

    dummy_initial_belief = BlocksPositionsBelief(n_blocks, workspace_x_lims_default, workspace_y_lims_default,
                                                 np.zeros((n_blocks, 2)), np.ones((n_blocks, 2)))
    
    if policy_type == "pouct":
        policy = POUCTPolicy(dummy_initial_belief, env.max_steps, goal_tower_position,
                             num_sims=planner_n_iterations,
                             stacking_reward=env.stacking_reward,
                             sensing_cost_coeff=env.sensing_cost_coeff,
                             stacking_cost_coeff=env.stacking_cost_coeff,
                             finish_ahead_of_time_reward_coeff=env.finish_ahead_of_time_reward_coeff,
                             max_planning_depth=max_planning_depth,
                             show_progress=True)
    elif policy_type == "hand_made":
        policy = HandMadePolicy(confidence_for_stack=confidence_for_stack)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
        
    
    experiment_mgr = ExperimentManager(env=env,
                                       policy=policy,
                                       visualize=True)

    reset_pile = True
    # Run experiments
    for _, row in df.iterrows():
        if pd.notna(row['conducted_datetime_stamp']) and row['conducted_datetime_stamp'] != '':
            continue  # Skip experiments that have already been conducted

        logging.info(f"Starting experiment ID: {row['experiment_id']}")

        experiment_mgr.run_from_list_and_save_results(row, config_file, results_dir)

        chime.success()

        reset_pile = True if not two_experiments_at_a_time else not reset_pile
        if reset_pile:
            input("Experiment completed. Make sure piles are reset and workspace is clean, "
                  "then Press ENTER to continue with the next experiment...")
            experiment_mgr.piles_manager.reset()

        experiment_mgr.visualizer.reset()

    experiment_mgr.stop_visualizer_if_started()
    logging.info("All experiments completed.")


if __name__ == "__main__":
    app()
