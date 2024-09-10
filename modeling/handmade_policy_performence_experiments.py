import typer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from typing import List
import time

from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform, workspace_x_lims_default, \
    workspace_y_lims_default, sample_block_positions_from_dists, goal_tower_position
from modeling.policies.hand_made_policy import HandMadePolicy
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.models.policy_model import BeliefModel
from modeling.pomdp_problem.domain.state import State
import pomdp_py


app = typer.Typer()


def run_single_experiment(n_blocks, max_steps, pickup_threshold, initial_belief, initial_state, tower_pos, experiment_params):
    agent = Agent(initial_blocks_position_belief=initial_belief,
                  max_steps=max_steps,
                  tower_position=tower_pos,
                  stacking_reward=experiment_params['stacking_reward'],
                  sensing_cost_coeff=experiment_params['sensing_cost_coeff'],
                  stacking_cost_coeff=experiment_params['stacking_cost_coeff'],
                  finish_ahead_of_time_reward_coeff=experiment_params['finish_ahead_of_time_reward_coeff'],
                  n_blocks_for_actions=experiment_params['n_blocks_for_actions'],
                  points_to_sample_for_each_block=experiment_params['points_to_sample_for_each_block'],
                  sensing_actions_to_sample_per_block=experiment_params['sensing_actions_to_sample_per_block'])

    env = Environment.from_agent(agent=agent, init_state=initial_state)

    policy = HandMadePolicy(confidence_for_stack=pickup_threshold)

    total_reward = 0

    for _ in range(max_steps):
        actual_action = policy.sample_action(agent.belief, agent.history)
        actual_reward = env.state_transition(actual_action, execute=True)
        actual_observation = env.provide_observation(agent.observation_model, actual_action)

        total_reward += actual_reward

        if isinstance(actual_observation, ObservationReachedTerminal):
            break

        agent.update(actual_action, actual_observation)

        belief = agent.belief
        if len(belief.block_beliefs) <= 0:
            # terminal state
            break

    return total_reward


@app.command()
def run_experiments(
        n_blocks: int = 4,
        max_steps: int = 20,
        stacking_reward: float = 1.0,
        sensing_cost_coeff: float = 0.05,
        stacking_cost_coeff: float = 0.05,
        finish_ahead_of_time_reward_coeff: float = 0.1,
        n_blocks_for_actions: int = 2,
        points_to_sample_for_each_block: int = 250,
        sensing_actions_to_sample_per_block: int = 2,
        max_planning_depth: int = 4,
        sigmin: float = 0.02,
        sigmax: float = 0.15,
        num_experiments: int = 25,
        num_runs_per_experiment: int = 2
):
    pickup_threshold = [0.5, 0.6, 0.7]
    tower_pos = goal_tower_position

    experiment_params = {
        'stacking_reward': stacking_reward,
        'sensing_cost_coeff': sensing_cost_coeff,
        'stacking_cost_coeff': stacking_cost_coeff,
        'finish_ahead_of_time_reward_coeff': finish_ahead_of_time_reward_coeff,
        'n_blocks_for_actions': n_blocks_for_actions,
        'points_to_sample_for_each_block': points_to_sample_for_each_block,
        'sensing_actions_to_sample_per_block': sensing_actions_to_sample_per_block,
        'max_planning_depth': max_planning_depth,
    }

    results = {ns: {'rewards': []} for ns in pickup_threshold}

    for exp in range(num_experiments):
        print(f"Running experiment {exp + 1}/{num_experiments}")

        # Sample initial belief
        mus = sample_block_positions_uniform(n_blocks, workspace_x_lims_default, workspace_y_lims_default)
        sigmas = np.random.uniform(sigmin, sigmax, size=(n_blocks, 2))
        initial_belief = BeliefModel(n_blocks, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

        # Sample initial state
        initial_block_positions = sample_block_positions_from_dists(initial_belief.block_beliefs)
        initial_state = State(steps_left=max_steps, block_positions=initial_block_positions, robot_position=tower_pos)

        for threshold in pickup_threshold:
            print(f"  Running with {threshold} threshold")
            exp_rewards = []

            for run in range(num_runs_per_experiment):
                reward = run_single_experiment(n_blocks, max_steps, threshold, initial_belief,
                                                              initial_state, tower_pos, experiment_params)
                exp_rewards.append(reward)

            results[threshold]['rewards'].extend(exp_rewards)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot rewards
    plot_results(pickup_threshold, results, 'rewards', 'Accumulated Reward', timestamp)

    # Save experiment parameters
    params = {
        'n_blocks': n_blocks,
        'max_steps': max_steps,
        'num_experiments': num_experiments,
        'num_runs_per_experiment': num_runs_per_experiment,
        'sigmin': sigmin,
        'sigmax': sigmax,
        **experiment_params
    }
    with open(f'results/params_{timestamp}.json', 'w') as f:
        json.dump(params, f, indent=2)


def plot_results(num_sims_list: List[int], results: dict, key: str, ylabel: str, timestamp: str):
    means = [np.mean(results[ns][key]) for ns in num_sims_list]
    std_devs = [np.std(results[ns][key]) for ns in num_sims_list]

    plt.figure(figsize=(10, 6))
    plt.errorbar(num_sims_list, means, yerr=std_devs, fmt='-o', capsize=5)
    plt.xlabel('Number of Simulations')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Number of Simulations')
    plt.grid(True)

    plt.savefig(f'results/{key}_{timestamp}.png')
    plt.close()


if __name__ == "__main__":
    app()