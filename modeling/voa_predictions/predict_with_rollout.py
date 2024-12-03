import os
import time
from copy import deepcopy
import typer
import pandas as pd
import numpy as np
from typing import List
import multiprocessing as mp

from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.pouct_planner_policy import POUCTPolicy
from modeling.policies.hand_made_policy import HandMadePolicy
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from modeling.voa_predictions.voa_prediction_rollout import predict_voa
from modeling.voa_predictions.utils import mujoco_camera_intrinsics
from modeling.voa_predictions.voa_prediction_rollout import rollout_episode, default_env_params, \
    predict_voa_with_sampled_states_parallel
from multi_mujoco.motion_planning.simulation_motion_planner import SimulationMotionPlanner
from experiments_sim.block_stacking_simulator import helper_camera_translation_from_ee

app = typer.Typer()


def load_or_create_results(results_file):
    """Load existing results or create new results file"""
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    return pd.DataFrame(columns=['row_id', 'empirical_voa', 'predicted_voa',
                                 'empirical_baseline', 'predicted_baseline', 'predicted_with_help'])


def save_results(results: List[dict], results_file: str):
    """Save results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)


def process_row(args):
    """Process a single row of data"""
    row, params = args

    # Setup environment
    motion_planner = SimulationMotionPlanner()
    gt = GeometryAndTransforms(
        motion_planner,
        cam_in_ee=-np.array(helper_camera_translation_from_ee)
    )

    # Initialize belief
    belief = BlocksPositionsBelief(
        len(eval(row['belief_mus'])),
        workspace_x_lims_default,
        workspace_y_lims_default,
        init_mus=eval(row['belief_mus']),
        init_sigmas=eval(row['belief_sigmas'])
    )

    # Create policy
    if params['policy_type'] == 'pouct':
        policy = POUCTPolicy(
            initial_belief=belief,
            num_sims=params['planner_sims'],
            max_planning_depth=params['planning_depth'],
            show_progress=False,
            **default_env_params
        )
    else:
        policy = HandMadePolicy()

    # Run prediction
    pred_voa, pred_no_help, pred_with_help = predict_voa(
        belief=belief,
        help_config=eval(row['help_config']),
        policy=policy,
        gt=gt,
        cam_intrinsic_matrix=mujoco_camera_intrinsics,
        n_states_to_sample=params['n_states'],
        times_repeat=params['times_repeat'],
        detection_probability_at_max_distance=params['detection_prob'],
        margin_in_pixels=params['margin_pixels'],
        detection_noise_scale=params['detection_noise'],
        n_processes=params['inner_processes']
    )

    return {
        'row_id': row.name,
        'empirical_voa': row['empirical_voa'],
        'predicted_voa': pred_voa,
        'empirical_baseline': row['empirical_baseline_value'],
        'predicted_baseline': pred_no_help,
        'predicted_with_help': pred_with_help
    }


@app.command()
def run_predictions(
        data_file: str = typer.Option(..., help="Input CSV file with experiment data"),
        results_file: str = typer.Option(..., help="Output CSV file for results"),
        policy_type: str = typer.Option("pouct", help="Policy type: 'pouct' or 'handmade'"),
        n_processes: int = typer.Option(4, help="Number of parallel processes for row processing"),
        inner_processes: int = typer.Option(1, help="Number of processes for each VOA prediction"),
        n_states: int = typer.Option(10, help="Number of states to sample for each prediction"),
        times_repeat: int = typer.Option(1, help="Number of times to repeat each prediction"),
        planner_sims: int = typer.Option(2000, help="Number of simulations for POUCT planner"),
        planning_depth: int = typer.Option(5, help="Planning depth for POUCT"),
        detection_prob: float = typer.Option(0.5, help="Detection probability at max distance"),
        margin_pixels: float = typer.Option(0.0, help="Margin in pixels"),
        detection_noise: float = typer.Option(0.3, help="Detection noise scale"),
        batch_size: int = typer.Option(50, help="Number of rows to process before saving"),
):
    """
    python -m modeling.voa_predictions.predict_with_rollout run-predictions --data-file="experiments_sim/results/experiments_planner_200_voa.csv" --results-file="modeling/voa_predictions/results/planner_200_10states.csv" --policy-type=pouct --n-processes=1 --inner-processes=20 --n-states=10 --times-repeat=1 --planner-sims=200 --planning-depth=4 --batch-size=5

    """
    # Verify that only one level of parallelization is used
    assert n_processes == 1 or inner_processes == 1, "Only one level of parallelization can be greater than 1"

    # Load data and existing results
    df = pd.read_csv(data_file)
    existing_results = load_or_create_results(results_file)

    # Find unprocessed rows
    processed_row_ids = set(existing_results['row_id'].values)
    unprocessed_df = df[~df.index.isin(processed_row_ids)]

    if len(unprocessed_df) == 0:
        print("All rows have been processed!")
        return

    print(f"Found {len(unprocessed_df)} unprocessed rows")

    # Setup parameters
    params = {
        'policy_type': policy_type,
        'n_states': n_states,
        'times_repeat': times_repeat,
        'planner_sims': planner_sims,
        'planning_depth': planning_depth,
        'detection_prob': detection_prob,
        'margin_pixels': margin_pixels,
        'detection_noise': detection_noise,
        'inner_processes': inner_processes,
    }

    # Process in batches
    all_results = existing_results.to_dict('records')

    # Choose processing method based on which level uses parallelization
    if n_processes > 1:
        # Outer parallelization
        for batch_start in range(0, len(unprocessed_df), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_df))
            batch_df = unprocessed_df.iloc[batch_start:batch_end]

            with mp.Pool(n_processes) as pool:
                batch_results = pool.map(
                    process_row,
                    [(row, params) for _, row in batch_df.iterrows()]
                )

            all_results.extend(batch_results)
            save_results(all_results, results_file)

            print_batch_metrics(batch_results, n_predictions=len(all_results) - len(existing_results),
                                total_predictions=len(unprocessed_df))

            if batch_end < len(unprocessed_df):
                time.sleep(1)
    else:
        # Inner parallelization (process rows sequentially)
        for batch_start in range(0, len(unprocessed_df), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_df))
            batch_df = unprocessed_df.iloc[batch_start:batch_end]

            batch_results = []
            for _, row in batch_df.iterrows():
                result = process_row((row, params))
                batch_results.append(result)

            all_results.extend(batch_results)
            save_results(all_results, results_file)

            print_batch_metrics(batch_results, n_predictions=len(all_results) - len(existing_results),
                                total_predictions=len(unprocessed_df))

            if batch_end < len(unprocessed_df):
                time.sleep(1)


def print_batch_metrics(batch_results, n_predictions, total_predictions):
    """Helper function to print metrics for a batch"""
    all_errors = [(r['predicted_voa'] - r['empirical_voa']) for r in batch_results]
    abs_errors = [abs(e) for e in all_errors]

    print(f"\nProcessed {n_predictions}/{total_predictions} new predictions")
    if len(all_errors) > 0:
        print(f"Batch metrics:")
        print(f"Mean error: {np.mean(all_errors):.3f}")
        print(f"Mean absolute error: {np.mean(abs_errors):.3f}")
        print(f"Over estimations (>0.3): {len([e for e in all_errors if e > 0.3])}")
        print(f"Under estimations (<-0.3): {len([e for e in all_errors if e < -0.3])}")


@app.command()
def run_from_code(
        data_file: str = typer.Option(..., help="Input CSV file with experiment data"),
        n_processes: int = typer.Option(4, help="Number of parallel processes for row processing"),
        inner_processes: int = typer.Option(1, help="Number of processes for each VOA prediction"),
):
    """Run multiple VOA prediction experiments with different parameters"""

    # Define parameter combinations to run
    experiments = [
        {
            'n_states': 10,
            'results_file': 'voa_predictions_n10_rollout.csv',
            'policy_type': 'pouct',
            'planner_sims': 200,
            'planning_depth': 4,
            'times_repeat': 1,
            'detection_prob': 0.5,
            'margin_pixels': 0.0,
            'detection_noise': 0.3,
            'batch_size': 10,
        },
        {
            'n_states': 30,
            'results_file': 'voa_predictions_n30_rollout.csv',
            'policy_type': 'pouct',
            'planner_sims': 200,
            'planning_depth': 4,
            'times_repeat': 1,
            'detection_prob': 0.5,
            'margin_pixels': 0.0,
            'detection_noise': 0.3,
            'batch_size': 10,
        },
        {
            'n_states': 50,
            'results_file': 'voa_predictions_n50_rollout.csv',
            'policy_type': 'pouct',
            'planner_sims': 200,
            'planning_depth': 4,
            'times_repeat': 1,
            'detection_prob': 0.5,
            'margin_pixels': 0.0,
            'detection_noise': 0.3,
            'batch_size': 10,
        }
    ]

    for params in experiments:
        print(f"\nStarting experiment with n_states={params['n_states']}")
        run_predictions(
            data_file=data_file,
            n_processes=n_processes,
            inner_processes=inner_processes,
            **params
        )
        print(f"Completed experiment with n_states={params['n_states']}\n")
        time.sleep(60)  # Wait a minute between experiments


if __name__ == "__main__":
    app()