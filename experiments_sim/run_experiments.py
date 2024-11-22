import gc
import os
import platform
import time
import csv
from copy import deepcopy
from functools import partial
import cv2
import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
import multiprocessing as mp
from experiments_lab.experiment_results_data import ExperimentResults
from experiments_lab.utils import plot_belief_with_history, update_belief
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from experiments_sim.position_estimation_service import create_position_estimation_service
from experiments_sim.utils import build_position_estimator, help_and_update_belief
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.vision.utils import detections_plots_no_depth_as_image
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from modeling.policies.pouct_planner_policy import POUCTPolicy
import logging
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default, goal_tower_position, \
    sample_block_positions_from_dists


app = typer.Typer()


class PositionEstimatorWrapper:
    def __init__(self,):
        dummy_env = BlockStackingSimulator(render_mode=None, visualize_mp=False,)
        gt = GeometryAndTransforms(dummy_env.motion_executor.motion_planner,
                                   cam_in_ee=-np.array(dummy_env.helper_camera_translation_from_ee))
        self.position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default,
                                                              gt,
                                                              "ur5e_1",
                                                              dummy_env.mujoco_env.get_robot_cam_intrinsic_matrix())

    def __call__(self, image, robot_config, plane_z, return_annotations=True, detect_on_cropped=True,
                 max_detections=5):
        return self.position_estimator.get_block_position_plane_projection(
            image, robot_config, plane_z=plane_z, return_annotations=return_annotations,
            detect_on_cropped=detect_on_cropped, max_detections=max_detections
        )


def _run_single_experiment(env: BlockStackingSimulator,
                           policy: AbastractPolicy,
                           init_block_positions,
                           init_block_belief: BlocksPositionsBelief,
                           help_config=None,
                           position_estimator_func: callable = None,
                           plot_belief=False, ):
    logging.info("EXPR running single experiment")

    results = ExperimentResults(policy_type=policy.__class__.__name__,
                                agent_params=policy.get_params(),
                                help_config=help_config)
    results.actual_initial_block_positions = init_block_positions
    results.belief_before_help = deepcopy(init_block_belief)
    actual_states = [env.mujoco_env.get_block_positions()]
    results.set_metadata("actual_states", actual_states)

    env.reset(init_block_positions)

    current_belief = deepcopy(init_block_belief)
    if help_config is not None:
        results.is_with_helper = True
        results.help_config = help_config
        results.belief_before_help = deepcopy(init_block_belief)

        position_estimator_func = position_estimator_func if position_estimator_func is not None\
            else PositionEstimatorWrapper()
        gt = GeometryAndTransforms(env.motion_executor.motion_planner,
                                   cam_in_ee=-np.array(env.helper_camera_translation_from_ee))
        detection_mus, detections_sigmas, detections_im = help_and_update_belief(env, current_belief, help_config,
                                                                                 position_estimator_func,
                                                                                 gt,
                                                                                 init_block_positions)
        help_observed_mus_and_sigmas = [(detection_mus[i], detections_sigmas[i]) for i in range(len(detection_mus))]

        results.help_detections_mus = detection_mus
        results.help_detections_sigmas = detections_sigmas

        if plot_belief:
            # the first belief that will be plotted is the one before the help
            actual_state_ = np.asarray(actual_states[0])
            actual_state_ = actual_state_[:, :2]  # take all blocks but just x and y
            plot_belief_with_history(results.belief_before_help, actual_state=actual_state_,
                                     observed_mus_and_sigmas=help_observed_mus_and_sigmas, ret_as_image=False)

            plt.figure(dpi=512, tight_layout=True, figsize=(5, 10))
            plt.imshow(detections_im)
            plt.axis('off')
            plt.show()
    else:
        results.is_with_helper = False
        help_observed_mus_and_sigmas, detections_im = None, None

    results.beliefs.append(current_belief)

    policy.reset(current_belief)
    history = []
    accumulated_reward = 0
    for i in range(env.max_steps):
        if plot_belief:
            actual_state_ = np.asarray(actual_states[-1])
            actual_state_ = actual_state_[:, :2]  # take all blocks but just x and y
            plot_belief_with_history(current_belief, actual_state=actual_state_,
                                     history=history, ret_as_image=False)

        action = policy.sample_action(current_belief, history)
        observation, reward = env.step(action)

        accumulated_reward += reward
        history.append((action, observation))

        current_belief = update_belief(current_belief, action, observation)

        results.beliefs.append(current_belief)
        results.actions.append(action)
        results.observations.append(observation)
        results.rewards.append(reward)
        actual_states.append(env.mujoco_env.get_block_positions())

        logging.info(f"EXPR step {i}. reward: {reward}, action: {action},  observation: {observation}, "
                     f"actual state: {actual_states[-1]}, accumulated reward: {accumulated_reward}")

        if isinstance(observation, ObservationReachedTerminal) or len(current_belief.block_beliefs) == 0 \
                or observation.steps_left <= 0:
            break

    results.total_reward = accumulated_reward

    logging.info(f"EXPR Experiment finished with total reward: {accumulated_reward}."
                 f" picked up {env.n_picked_blocks} blocks")

    return results, help_observed_mus_and_sigmas, detections_im


@app.command(
    context_settings={"ignore_unknown_options": True})
def run_single_experiment_with_planner(max_steps: int = 20,
                                       seed: int = typer.Option(42, help="for sampling initial belief and state"),
                                       min_prior_std: float = typer.Option(0.03,
                                                                           help="min prior std for block positions"),
                                       max_prior_std: float = typer.Option(0.15,
                                                                           help="max prior std for block positions"),
                                       n_sims: int = typer.Option(2000, help="number of simulations for POUCT planner"),
                                       max_planning_depth: int = typer.Option(5, help="max planning depth for POUCT"
                                                                                      " planner both for rollout and tree"),
                                       help_config_idx: int = typer.Option(-1, help="index of the help config to use"
                                                                                    "from the file. -1 for no help"),
                                       render_env: bool = typer.Option(True, help="whether to render the environment"),
                                       real_time_rendering: bool = typer.Option(True,
                                                                                help="whether to wait when rendiring to maintain"
                                                                                     "real time or go as fast as possible"),
                                       visualize_mp: bool = typer.Option(False,
                                                                         help="whether to visualize the motion planner"),
                                       show_planner_progress: bool = typer.Option(True,
                                                                                  help="whether to show progress of task planner"),
                                       plot_belief: bool = typer.Option(True,
                                                                        help="whether to plot belief at each step"),
                                       results_dir: str = typer.Option("", help="directory to save results"
                                                                                "will be saved in an internal dir with datetime stamp"), ):
    np.random.seed(seed)
    n_blocks = 4  # maybe parameter in the future

    if results_dir == "":
        results_dir = os.path.join(os.path.dirname(__file__), "experiments")
    datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(results_dir, datetime_stamp)
    os.makedirs(experiment_dir, exist_ok=True)

    logging.info(f"Experiment dir: {experiment_dir}")

    help_configs = np.load(os.path.join(os.path.dirname(__file__), "configurations/sim_help_configs_40.npy"))
    help_config = help_configs[help_config_idx] if help_config_idx >= 0 else None

    render_mode = "human" if render_env else None
    env = BlockStackingSimulator(max_steps=max_steps, render_mode=render_mode,
                                 render_sleep_to_maintain_fps=real_time_rendering,
                                 visualize_mp=visualize_mp)

    block_pos_mu = np.random.uniform(low=[workspace_x_lims_default[0], workspace_y_lims_default[0]],
                                     high=[workspace_x_lims_default[1], workspace_y_lims_default[1]],
                                     size=(n_blocks, 2))
    block_pos_sigma = np.random.uniform(low=min_prior_std, high=max_prior_std, size=(n_blocks, 2))
    init_block_belief = BlocksPositionsBelief(n_blocks, workspace_x_lims_default, workspace_y_lims_default,
                                              init_mus=block_pos_mu, init_sigmas=block_pos_sigma)
    init_block_positions = sample_block_positions_from_dists(init_block_belief.block_beliefs, min_dist=0.08)

    policy = POUCTPolicy(initial_belief=init_block_belief, max_steps=max_steps, tower_position=goal_tower_position,
                         max_planning_depth=max_planning_depth, num_sims=n_sims, show_progress=show_planner_progress,
                         stacking_reward=env.stacking_reward, sensing_cost_coeff=env.sensing_cost_coeff,
                         stacking_cost_coeff=env.stacking_cost_coeff,
                         finish_ahead_of_time_reward_coeff=env.finish_ahead_of_time_reward_coeff)

    position_estimator_func = PositionEstimatorWrapper()

    results, help_obs_mus_and_sigmas, help_plot = _run_single_experiment(env, policy, init_block_positions,
                                                                         init_block_belief, help_config=help_config,
                                                                         position_estimator_func=position_estimator_func,
                                                                         plot_belief=plot_belief)

    results.save(os.path.join(experiment_dir, "results.pkl"))
    if help_plot is not None:
        cv2.imwrite(os.path.join(experiment_dir, "help_detections_im.png"), cv2.cvtColor(help_plot, cv2.COLOR_RGB2BGR))
        # belief before help:
        b_before_help_im = plot_belief_with_history(results.belief_before_help,
                                                    actual_state=results.actual_initial_block_positions,
                                                    observed_mus_and_sigmas=help_obs_mus_and_sigmas, ret_as_image=True)
        b_after_help_im = plot_belief_with_history(results.beliefs[0],
                                                   actual_state=results.actual_initial_block_positions,
                                                   observed_mus_and_sigmas=help_obs_mus_and_sigmas, ret_as_image=True)
        b_after_help_im = cv2.cvtColor(b_after_help_im, cv2.COLOR_RGB2BGR)
        b_before_help_im = cv2.cvtColor(b_before_help_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(experiment_dir, "belief_before_help_im.png"), b_before_help_im)
        cv2.imwrite(os.path.join(experiment_dir, "belief_after_help_im.png"), b_after_help_im)


class SharedPositionEstimatorClient:
    """Client for the position estimation service"""
    def __init__(self, service):
        self.service = service

    def __call__(self, image, robot_config, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                image = np.ascontiguousarray(image, dtype=np.uint8)
                robot_config = np.asarray(robot_config, dtype=np.float64)
                response = self.service.estimate_position(image, robot_config)
                if response is None:
                    raise RuntimeError("Position estimation failed")
                return response
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)


def run_experiment_wrapper(args):
    kwargs_dict, results_dir, position_service = args

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    process_id = os.getpid()
    date_time_pid_stamp = f"{datetime_stamp}_pid{process_id}"
    experiment_dir = os.path.join(results_dir, date_time_pid_stamp)
    os.makedirs(experiment_dir, exist_ok=True)

    row = kwargs_dict['row']

    # Create environment
    env = BlockStackingSimulator(
        max_steps=kwargs_dict['max_steps'],
        render_mode=kwargs_dict.get('render_mode', None),
        render_sleep_to_maintain_fps=kwargs_dict.get('real_time_rendering', False),
        visualize_mp=kwargs_dict.get('visualize_mp', False)
    )

    # Initialize belief from CSV data
    init_block_belief = BlocksPositionsBelief(
        4,  # n_blocks is fixed to 4 as per your example
        workspace_x_lims_default,
        workspace_y_lims_default,
        init_mus=np.array(eval(row['belief_mus'])),
        init_sigmas=np.array(eval(row['belief_sigmas']))
    )
    init_block_positions = np.array(eval(row['state']))

    # Parse help config from row
    if pd.isna(row['help_config']) or row['help_config'] == '' or row['help_config'] == 'None':
        help_config = None
    else:
        help_config = np.array(eval(row['help_config']))

    # Create policy
    policy = POUCTPolicy(
        initial_belief=init_block_belief,
        max_steps=kwargs_dict['max_steps'],
        tower_position=goal_tower_position,
        max_planning_depth=kwargs_dict['max_planning_depth'],
        num_sims=kwargs_dict['n_sims'],
        show_progress=kwargs_dict.get('show_planner_progress', False),
        stacking_reward=env.stacking_reward,
        sensing_cost_coeff=env.sensing_cost_coeff,
        stacking_cost_coeff=env.stacking_cost_coeff,
        finish_ahead_of_time_reward_coeff=env.finish_ahead_of_time_reward_coeff
    )

    position_estimator_func = SharedPositionEstimatorClient(position_service)

    try:
        results, help_obs_mus_and_sigmas, help_plot = _run_single_experiment(
            env=env,
            policy=policy,
            init_block_positions=init_block_positions,
            init_block_belief=init_block_belief,
            help_config=help_config,
            plot_belief=kwargs_dict.get('plot_belief', False),
            position_estimator_func=position_estimator_func
        )

        # Save results and plots
        results.set_metadata('experiment_id', row['experiment_id'])
        results.set_metadata('belief_idx', row['belief_idx'])
        results.set_metadata('state_idx', row['state_idx'])
        results.set_metadata('help_config_idx_local', row['help_config_idx_local'])
        results.save(os.path.join(experiment_dir, "results.pkl"))

        if help_plot is not None:
            cv2.imwrite(os.path.join(experiment_dir, "help_detections_im.png"),
                        cv2.cvtColor(help_plot, cv2.COLOR_RGB2BGR))

        return row['experiment_id'],  date_time_pid_stamp, True

    except Exception as e:
        print(f"Process {process_id} failed with error: {str(e)}")
        return row['experiment_id'], None, False
    finally:
        plt.close('all')
        gc.collect()


@app.command()
def run_experiments_from_list_parallel(
        n_processes: int = 2,
        experiments_file: str = typer.Option("experiments_sim/configurations/experiments_planner_2000.csv",
                                             help="file with list of experiment to draw from and write to"),
        max_steps: int = 20,
        n_sims: int = typer.Option(2000, help="number of simulations for POUCT planner"),
        max_planning_depth: int = typer.Option(5,
                                               help="max planning depth for POUCT planner both for rollout and tree"),
        render_env: bool = typer.Option(False, help="whether to render the environment"),
        real_time_rendering: bool = typer.Option(False,
                                                 help="whether to wait when rendering to maintain real time or go as fast as possible"),
        visualize_mp: bool = typer.Option(False, help="whether to visualize the motion planner"),
        show_planner_progress: bool = typer.Option(False, help="whether to show progress of task planner"),
        plot_belief: bool = typer.Option(False, help="whether to plot belief at each step"),
        results_dir: str = typer.Option("",
                                        help="directory to save results will be saved in an internal dir with datetime stamp"),
):
    if platform.system() == 'Linux':
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    if results_dir == "":
        results_dir = os.path.join(os.path.dirname(__file__), "experiments")

    position_service = create_position_estimation_service()

    try:
        # Read experiments file
        df = pd.read_csv(experiments_file)
        logging.info(f"Loaded {len(df)} experiments from {experiments_file}")

        # Filter undone experiments
        undone_experiments = df[pd.isna(df['conducted_datetime_stamp']) | (df['conducted_datetime_stamp'] == '')]
        n_init_undone = len(undone_experiments)

        if n_init_undone == 0:
            logging.info("No experiments left to run")
            return

        def update_csv(experiment_id, experiment_dir):
            """Safely update CSV with completion status and directory"""
            df = pd.read_csv(experiments_file)
            idx = df['experiment_id'] == experiment_id
            df.loc[idx, 'conducted_datetime_stamp'] = experiment_dir
            df.to_csv(experiments_file, index=False)
            del df

        experiment_args = []
        for _, row in undone_experiments.iterrows():
            kwargs = {
                'max_steps': max_steps,
                'render_mode': "human" if render_env else None,
                'real_time_rendering': real_time_rendering,
                'visualize_mp': visualize_mp,
                'n_sims': n_sims,
                'max_planning_depth': max_planning_depth,
                'show_planner_progress': show_planner_progress,
                'plot_belief': plot_belief,
                'row': row
            }
            experiment_args.append((kwargs, results_dir, position_service))

        n_done = 0
        with mp.Pool(n_processes) as pool:
            for result in pool.imap_unordered(run_experiment_wrapper, experiment_args):
                experiment_id, experiment_dir, success = result
                if success:
                    update_csv(experiment_id, experiment_dir)
                    logging.info(f"Completed and updated experiment {experiment_id}")
                    n_done += 1
                    print(f"---Completed {n_done}/{n_init_undone} experiments in this run---")
                else:
                    logging.error(f"Failed experiment {experiment_id}")

                if n_done % 50 == 0:
                    gc.collect()

    finally:
        position_service.shutdown()

    logging.info("All experiments completed")


if __name__ == '__main__':
    app()
