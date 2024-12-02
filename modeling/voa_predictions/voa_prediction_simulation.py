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
                                    print_progress=False) -> (float, float, float):
    """
    Predict the VOA based on the given belief, help_config, policy and use the already sampled states
    @param belief:
    @param help_config:
    @param policy:
    @param states: list of block positions listsb
    @param states_likelihoods: list of likelihoods for the states (what is the likelihood of each state in the belief)
    @param gt: GeometryAndTransforms object for the scene
    @param cam_intrinsic_matrix: camera intrinsic matrix
    @param print_progress:
    @return: tuple of (voa_pred, pred_value_no_help, pred_value_with_help)
    """

    states_value_diffs = []
    states_values_no_help = []
    states_values_with_help = []
    for i, s in enumerate(states):
        print(f'Simulating state {i + 1}/{len(states)}') if print_progress else None

        belief_with_help = deepcopy(belief)
        observed_detection_mus, observed_detection_sigmas = sample_observation_fov_based(
            s, help_config, gt, cam_intrinsic_matrix)
        if len(observed_detection_mus) == 0:
            states_value_diffs.append(0)
            continue
        belief_with_help.update_from_image_detections_position_distribution(
            observed_detection_mus, observed_detection_sigmas)

        reward_with_help = rollout_episode(belief_with_help, policy, s)
        reward_no_help = rollout_episode(deepcopy(belief), policy, s)

        states_values_no_help.append(reward_no_help)
        states_values_with_help.append(reward_with_help)
        states_value_diffs.append(reward_with_help - reward_no_help)

    weights = np.asarray(states_likelihoods)
    weights /= np.sum(weights)

    return np.dot(states_value_diffs, weights)[0],\
        np.dot(states_values_no_help, weights)[0],\
        np.dot(states_values_with_help, weights)[0]


def predict_voa_with_sampled_states_parallel(belief: BlocksPositionsBelief, help_config, policy: AbstractPolicy,
                                             states, states_likelihoods, gt: GeometryAndTransforms,
                                             cam_intrinsic_matrix,
                                             print_progress=False, n_processes=None) -> float:
    """
    Predict the VOA based on the given belief, help_config, policy and use the already sampled states
    Now with parallel processing support
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    states_value_diffs = []
    valid_states = []
    valid_likelihoods = []

    # Process the observations sequentially
    for i, s in enumerate(states):
        print(f'Processing observations for state {i + 1}/{len(states)}') if print_progress else None

        observed_detection_mus, observed_detection_sigmas = sample_observation_fov_based(
            s, help_config, gt, cam_intrinsic_matrix)

        if len(observed_detection_mus) == 0:
            states_value_diffs.append(0)
            continue

        belief_with_help = deepcopy(belief)
        belief_with_help.update_from_image_detections_position_distribution(
            observed_detection_mus, observed_detection_sigmas)

        valid_states.append((s, belief_with_help))
        valid_likelihoods.append(states_likelihoods[i])

    if print_progress:
        print(f'Running parallel rollouts with {n_processes} processes')

    with mp.Pool(processes=n_processes) as pool:
        # Run both rollouts for each state in parallel
        results = pool.starmap(rollout_episode,
                               [(belief_copy, policy, state) for state, belief_copy in valid_states] +  # with help
                               [(belief, policy, state) for state, _ in valid_states])  # without help

        # Split results into with_help and without_help
        n = len(valid_states)
        rewards_with_help = results[:n]
        rewards_without_help = results[n:]

        # Calculate differences
        states_value_diffs.extend([r1 - r2 for r1, r2 in zip(rewards_with_help, rewards_without_help)])

    weights = np.asarray(states_likelihoods)
    weights /= np.sum(weights)

    return np.dot(states_value_diffs, weights)[0]
