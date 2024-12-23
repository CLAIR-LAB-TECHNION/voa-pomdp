import numpy as np

from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbstractPolicy
from modeling.voa_predictions.observation_prediction_functions import sample_observation_fov_based
from copy import deepcopy
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.models.policy_model import BeliefModel
from modeling.pomdp_problem.domain.state import State
from experiments_lab.utils import update_belief
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from modeling.policies.pouct_planner_policy import POUCTPolicy
from frozendict import frozendict
import multiprocessing as mp
from functools import partial
from lab_ur_stack.utils.workspace_utils import sample_block_positions_from_dists


default_env_params = {
    'max_steps': 20,
    'tower_position': [-0.45, -1.15],
    'stacking_reward': 1,
    'sensing_cost_coeff': 0.05,
    'stacking_cost_coeff': 0.05,
    'finish_ahead_of_time_reward_coeff': 0.1,
}
# make it immutable
default_env_params = frozendict(default_env_params)


def rollout_episode(belief: BlocksPositionsBelief, policy: AbstractPolicy, state,
                    env_params=default_env_params) -> float:
    """
    Rollout an episode with the given belief, initial state and policy and return prediction for accumulated reward
    @param belief:
    @param help_config:
    @param policy:
    @param env_params:
    @param state:
    @return: accumulated reward
    """
    max_steps = env_params['max_steps']

    # change to pomdp_py data structures
    belief = BeliefModel.from_block_positions_belief(belief)
    model_state = State(block_positions=state,
                        robot_position=env_params['tower_position'],
                        steps_left=max_steps)

    agent = Agent(initial_blocks_position_belief=belief,
                  max_steps=env_params['max_steps'],
                  tower_position=env_params['tower_position'],
                  stacking_reward=env_params['stacking_reward'],
                  sensing_cost_coeff=env_params['sensing_cost_coeff'],
                  stacking_cost_coeff=env_params['stacking_cost_coeff'],
                  finish_ahead_of_time_reward_coeff=env_params['finish_ahead_of_time_reward_coeff'], )

    env = Environment.from_agent(agent=agent, init_state=model_state)

    current_belief = deepcopy(belief)
    total_reward = 0
    history = []
    policy.reset(current_belief)
    for _ in range(max_steps):
        actual_action = policy.sample_action(current_belief, history)
        actual_reward = env.state_transition(actual_action, execute=True)
        actual_observation = env.provide_observation(agent.observation_model, actual_action)

        total_reward += actual_reward
        history.append((actual_action, actual_observation))

        if isinstance(actual_observation, ObservationReachedTerminal):
            break

        current_belief = update_belief(current_belief, actual_action, actual_observation)

        if len(current_belief.block_beliefs) <= 0:
            break

    return total_reward


def predict_voa_with_sampled_states(belief: BlocksPositionsBelief, help_config, policy: AbstractPolicy,
                                    states, states_likelihoods, gt: GeometryAndTransforms, cam_intrinsic_matrix,
                                    detection_probability_at_max_distance=0.9, max_distance=1.5, margin_in_pixels=30,
                                    detection_noise_scale=0.1, print_progress=False) -> (float, float, float):
    """
    Predict the VOA based on the given belief, help_config, policy and use the already sampled states
    @param belief: BlocksPositionsBelief representing current belief state
    @param help_config: Configuration of helper robot
    @param policy: Policy to evaluate
    @param states: list of block positions lists
    @param states_likelihoods: list of likelihoods for the states (what is the likelihood of each state in the belief)
    @param gt: GeometryAndTransforms object for the scene
    @param cam_intrinsic_matrix: camera intrinsic matrix
    @param detection_probability_at_max_distance: Detection probability at and beyond max_distance
    @param max_distance: Maximum distance threshold in meters
    @param margin_in_pixels: Blocks within this margin from image edges are considered out of FOV
    @param print_progress: Whether to print progress information
    @return: tuple of (voa_pred, pred_value_no_help, pred_value_with_help)
    """

    states_value_diffs = []
    states_values_no_help = []
    states_values_with_help = []
    for i, s in enumerate(states):
        print(f'Simulating state {i + 1}/{len(states)}') if print_progress else None

        reward_no_help = rollout_episode(deepcopy(belief), policy, s)

        belief_with_help = deepcopy(belief)
        observed_detection_mus, observed_detection_sigmas = sample_observation_fov_based(
            s, help_config, gt, cam_intrinsic_matrix,
            detection_probability_at_max_distance=detection_probability_at_max_distance,
            max_distance=max_distance,
            margin_in_pixels=margin_in_pixels,
            detection_noise_scale=detection_noise_scale)

        if len(observed_detection_mus) == 0:
            states_value_diffs.append(0)
            states_values_no_help.append(reward_no_help)
            states_values_with_help.append(reward_no_help)
            continue

        belief_with_help.update_from_image_detections_position_distribution(
            observed_detection_mus, observed_detection_sigmas)

        reward_with_help = rollout_episode(belief_with_help, policy, s)

        states_values_no_help.append(reward_no_help)
        states_values_with_help.append(reward_with_help)
        states_value_diffs.append(reward_with_help - reward_no_help)

    weights = np.asarray(states_likelihoods)
    weights /= np.sum(weights)

    return np.dot(states_value_diffs, weights)[0], \
        np.dot(states_values_no_help, weights)[0], \
        np.dot(states_values_with_help, weights)[0]


def predict_voa_with_sampled_states_parallel(belief: BlocksPositionsBelief, help_config, policy: AbstractPolicy,
                                             states, states_likelihoods, gt: GeometryAndTransforms,
                                             cam_intrinsic_matrix,
                                             detection_probability_at_max_distance=0.9, max_distance=1.5,
                                             margin_in_pixels=30, detection_noise_scale=0.1,
                                             print_progress=False, n_processes=2,) -> (float, float, float):
    """
    Parallel version of predict_voa_with_sampled_states
    @param belief: BlocksPositionsBelief representing current belief state
    @param help_config: Configuration of helper robot
    @param policy: Policy to evaluate
    @param states: list of block positions lists
    @param states_likelihoods: list of likelihoods for the states
    @param gt: GeometryAndTransforms object for the scene
    @param cam_intrinsic_matrix: camera intrinsic matrix
    @param detection_probability_at_max_distance: Detection probability at and beyond max_distance
    @param max_distance: Maximum distance threshold in meters
    @param margin_in_pixels: Blocks within this margin from image edges are considered out of FOV
    @param print_progress: Whether to print progress information
    @param n_processes: Number of parallel processes to use
    @param detection_noise_scale: Scale factor for detection position noise
    @return: tuple of (voa_pred, pred_value_no_help, pred_value_with_help)
    """
    if n_processes == 1:
        args = locals().copy()
        del args['n_processes']
        return predict_voa_with_sampled_states(**args)

    from collections import namedtuple
    ObservedState = namedtuple('ObservedState', ['state', 'belief_with_help'])

    # Process observations sequentially and prepare states for parallel rollouts
    states_for_rollout = []  # Will contain None for no-observation states, or ObservedState for states with observations

    for i, state in enumerate(states):
        print(f'Processing observations for state {i + 1}/{len(states)}') if print_progress else None

        observed_detection_mus, observed_detection_sigmas = sample_observation_fov_based(
            state, help_config, gt, cam_intrinsic_matrix,
            detection_probability_at_max_distance=detection_probability_at_max_distance,
            max_distance=max_distance,
            margin_in_pixels=margin_in_pixels,
            detection_noise_scale=detection_noise_scale)

        if len(observed_detection_mus) == 0:
            states_for_rollout.append(None)
            continue

        belief_with_help = deepcopy(belief)
        belief_with_help.update_from_image_detections_position_distribution(
            observed_detection_mus, observed_detection_sigmas)
        states_for_rollout.append(ObservedState(state, belief_with_help))

    if print_progress:
        print(f'Running parallel rollouts with {n_processes} processes')

    # Prepare parallel tasks
    parallel_tasks = []

    for i, state_info in enumerate(states_for_rollout):
        if state_info is None:
            # For no-observation states, just one rollout without help
            parallel_tasks.append((deepcopy(belief), deepcopy(policy), states[i]))
        else:
            # For states with observations, add both with-help and without-help rollouts
            parallel_tasks.append((state_info.belief_with_help, deepcopy(policy), state_info.state))
            parallel_tasks.append((deepcopy(belief), deepcopy(policy), state_info.state))

    # Run all rollouts in parallel
    with mp.Pool(processes=n_processes) as pool:
        all_rewards = pool.starmap(rollout_episode, parallel_tasks)

    # Process results into final format
    final_results = []
    reward_idx = 0

    for state_info in states_for_rollout:
        if state_info is None:
            # For no-observation states, reward is the same for both cases
            reward = all_rewards[reward_idx]
            final_results.append((0, reward, reward))
            reward_idx += 1
        else:
            # For states with observations, get both rewards
            reward_with_help = all_rewards[reward_idx]
            reward_no_help = all_rewards[reward_idx + 1]
            final_results.append((reward_with_help - reward_no_help,
                                  reward_no_help,
                                  reward_with_help))
            reward_idx += 2

    # Calculate weighted averages
    value_diffs, values_no_help, values_with_help = zip(*final_results)
    weights = np.asarray(states_likelihoods)
    weights /= np.sum(weights)

    return (np.dot(value_diffs, weights)[0],
            np.dot(values_no_help, weights)[0],
            np.dot(values_with_help, weights)[0])



def predict_voa(belief: BlocksPositionsBelief, help_config, policy: AbstractPolicy,
                gt: GeometryAndTransforms, cam_intrinsic_matrix, n_states_to_sample, times_repeat=1,
                detection_probability_at_max_distance=0.9, max_distance=1.5, margin_in_pixels=30,
                detection_noise_scale=0.1, n_processes=1, print_progress=False) -> (float, float, float):
    states = [sample_block_positions_from_dists(belief.block_beliefs) for _ in range(n_states_to_sample)]
    states_likelihoods = [belief.state_pdf(state) for state in states]
    pred_voa = []
    pred_no_help = []
    pred_with_help = []

    for _ in range(times_repeat):
        voa, no_help, with_help = (
            predict_voa_with_sampled_states_parallel(belief, help_config, policy, states, states_likelihoods,
                                                    gt, cam_intrinsic_matrix, n_processes=n_processes,
                                                    detection_probability_at_max_distance=detection_probability_at_max_distance,
                                                    max_distance=max_distance, margin_in_pixels=margin_in_pixels,
                                                    detection_noise_scale=detection_noise_scale,
                                                     print_progress=print_progress))
        pred_voa.append(voa)
        pred_no_help.append(no_help)
        pred_with_help.append(with_help)

    return np.mean(pred_voa), np.mean(pred_no_help), np.mean(pred_with_help)