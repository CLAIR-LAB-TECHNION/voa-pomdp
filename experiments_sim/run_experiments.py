import os
import time
from copy import deepcopy

import cv2
import numpy as np
import typer
from matplotlib import pyplot as plt

from experiments_lab.experiment_results_data import ExperimentResults
from experiments_lab.utils import plot_belief_with_history, update_belief
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from experiments_sim.utils import build_position_estimator, help_and_update_belief
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from modeling.policies.pouct_planner_policy import POUCTPolicy
import logging
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default, goal_tower_position, \
    sample_block_positions_from_dists

app = typer.Typer()


def _run_single_experiment(env: BlockStackingSimulator,
                           policy: AbastractPolicy,
                           init_block_positions,
                           init_block_belief: BlocksPositionsBelief,
                           help_config=None,
                           position_estimator=None,
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

        position_estimator = position_estimator if position_estimator is not None else build_position_estimator(env)[0]
        detection_mus, detections_sigmas, detections_im = help_and_update_belief(env, current_belief, help_config,
                                                                                 position_estimator,
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

    help_configs = np.load(os.path.join(os.path.dirname(__file__), "sim_help_configs.npy"))
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

    position_estimator, _ = build_position_estimator(env)

    results, help_obs_mus_and_sigmas, help_plot = _run_single_experiment(env, policy, init_block_positions,
                                                                         init_block_belief, help_config=help_config,
                                                                         position_estimator=position_estimator,
                                                                         plot_belief=plot_belief)

    results.save(os.path.join(experiment_dir, "results.pkl"))
    if help_plot is not None:
        cv2.imwrite(os.path.join(experiment_dir, "help_detections_im.png"), help_plot)
        # belief before help:
        b_before_help_im = plot_belief_with_history(results.belief_before_help,
                                                    actual_state=results.actual_initial_block_positions,
                                                    observed_mus_and_sigmas=help_obs_mus_and_sigmas, ret_as_image=True)
        b_after_help_im = plot_belief_with_history(results.beliefs[0],
                                                   actual_state=results.actual_initial_block_positions,
                                                   observed_mus_and_sigmas=help_obs_mus_and_sigmas, ret_as_image=True)
        cv2.imwrite(os.path.join(experiment_dir, "belief_before_help_im.png"), b_before_help_im)
        cv2.imwrite(os.path.join(experiment_dir, "belief_after_help_im.png"), b_after_help_im)
    # TODO rgb2bgr?

    pass


if __name__ == '__main__':
    app()
