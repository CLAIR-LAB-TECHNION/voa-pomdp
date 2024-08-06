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
from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.models.policy_model import BeliefModel
from modeling.pomdp_problem.domain.state import State
import pomdp_py


app = typer.Typer()


def run_single_experiment(n_blocks, max_steps, num_sims, initial_belief, initial_state, tower_pos, experiment_params):
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

    planner = pomdp_py.POUCT(max_depth=experiment_params['max_planning_depth'],
                             num_sims=num_sims,
                             discount_factor=1.0,
                             rollout_policy=agent.policy_model,
                             show_progress=False)

    total_reward = 0
    planning_times = []

    for _ in range(max_steps):
        actual_action = planner.plan(agent)
        planning_time = planner.last_planning_time
        planning_times.append(planning_time)

        actual_reward = env.state_transition(actual_action, execute=True)
        actual_observation = env.provide_observation(agent.observation_model, actual_action)

        total_reward += actual_reward

        if isinstance(actual_observation, ObservationReachedTerminal):
            break

        planner.update(agent, actual_action, actual_observation)  # updates tree but not belief
        agent.update(actual_action, actual_observation)

        belief = agent.belief
        if len(belief.block_beliefs) <= 0:
            # terminal state
            break

    return total_reward, np.mean(planning_times)


@app.command()
def run_experiments(
        n_blocks: int = 3,
        max_steps: int = 20,
        stacking_reward: float = 1.0,
        sensing_cost_coeff: float = 0.1,
        stacking_cost_coeff: float = 0.2,
        finish_ahead_of_time_reward_coeff: float = 0.1,
        n_blocks_for_actions: int = 2,
        points_to_sample_for_each_block: int = 150,
        sensing_actions_to_sample_per_block: int = 2,
        max_planning_depth: int = 6,
        sigmin: float = 0.01,
        sigmax: float = 0.15,
        num_experiments: int = 1,
        num_runs_per_experiment: int = 10
):
    num_sims_list = [500, 1000, 2000, 5000, 10000]
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

    results = {ns: {'rewards': [], 'planning_times': []} for ns in num_sims_list}

    for exp in range(num_experiments):
        print(f"Running experiment {exp + 1}/{num_experiments}")

        # Sample initial belief
        mus = sample_block_positions_uniform(n_blocks, workspace_x_lims_default, workspace_y_lims_default)
        sigmas = np.random.uniform(sigmin, sigmax, size=(n_blocks, 2))
        initial_belief = BeliefModel(n_blocks, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

        # Sample initial state
        initial_block_positions = sample_block_positions_from_dists(initial_belief.block_beliefs)
        initial_state = State(steps_left=max_steps, block_positions=initial_block_positions, robot_position=tower_pos)

        for num_sims in num_sims_list:
            print(f"  Running with {num_sims} simulations")
            exp_rewards = []
            exp_planning_times = []

            for run in range(num_runs_per_experiment):
                reward, planning_time = run_single_experiment(n_blocks, max_steps, num_sims, initial_belief,
                                                              initial_state, tower_pos, experiment_params)
                exp_rewards.append(reward)
                exp_planning_times.append(planning_time)

            results[num_sims]['rewards'].extend(exp_rewards)
            results[num_sims]['planning_times'].extend(exp_planning_times)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot rewards
    plot_results(num_sims_list, results, 'rewards', 'Accumulated Reward', timestamp)

    # Plot planning times
    plot_results(num_sims_list, results, 'planning_times', 'Average Planning Time per Step (s)', timestamp)

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