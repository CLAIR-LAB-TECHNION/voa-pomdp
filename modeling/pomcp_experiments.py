import typer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from typing import List
import time
from multiprocessing import Pool, cpu_count

from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform, workspace_x_lims_default, \
    workspace_y_lims_default, sample_block_positions_from_dists, goal_tower_position
from modeling.belief.particle_belief_utils import explicit_belief_to_particles
from modeling.policies.pomcp_policy import POMCPPolicy
from modeling.pomdp_problem.agent.agent_v2 import AgentV2
from modeling.pomdp_problem.domain.action import DummyAction
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.domain.state import State
import pomdp_py
from modeling.belief.block_position_belief import BlocksPositionsBelief

app = typer.Typer()


def run_single_experiment(n_blocks, max_steps, num_sims, initial_belief, initial_state, tower_pos,
                          max_planning_depth, experiment_params, show_progress=False):
    policy = POMCPPolicy(initial_belief=initial_belief,
                         max_steps=max_steps,
                         tower_position=tower_pos,
                         max_planning_depth=max_planning_depth,
                         num_sims=num_sims,
                         stacking_reward=experiment_params['stacking_reward'],
                         sensing_cost_coeff=experiment_params['sensing_cost_coeff'],
                         stacking_cost_coeff=experiment_params['stacking_cost_coeff'],
                         finish_ahead_of_time_reward_coeff=experiment_params['finish_ahead_of_time_reward_coeff'],
                         n_sensing_actions_to_sample_per_block=experiment_params[
                             'n_sensing_actions_to_sample_per_block'],
                         n_pickup_actions_to_sample_per_block=experiment_params['n_pickup_actions_to_sample_per_block'],
                         action_points_std=experiment_params['action_points_std'],
                         show_progress=show_progress)

    env = Environment.from_agent(agent=policy.agent, init_state=initial_state)
    policy.reset(initial_belief)

    total_reward = 0
    planning_times = []

    for _ in range(max_steps):
        actual_action = policy.sample_action()
        planning_time = policy.planner.last_planning_time
        planning_times.append(planning_time)

        actual_reward = env.state_transition(actual_action, execute=True)
        actual_observation = env.provide_observation(policy.agent.observation_model, actual_action)

        total_reward += actual_reward

        if isinstance(actual_observation, ObservationReachedTerminal) or isinstance(actual_action, DummyAction):
            break

        policy.update(actual_action, actual_observation)
        cur_belief = policy.get_belief()
        # TODO: do something with it.

    return total_reward, np.mean(planning_times)


@app.command()
def test_single_experiment():
    n_blocks = 4
    max_steps = 20
    num_sims = 5000
    max_planning_depth = 10
    tower_pos = goal_tower_position
    num_particles = 20000

    experiment_params = {
        'stacking_reward': 1.0,
        'sensing_cost_coeff': 0.05,
        'stacking_cost_coeff': 0.05,
        'finish_ahead_of_time_reward_coeff': 0.1,
        'n_sensing_actions_to_sample_per_block': 3,
        'n_pickup_actions_to_sample_per_block': 3,
        'action_points_std': 0.1
    }

    np.random.seed(42)
    mus = sample_block_positions_uniform(n_blocks, workspace_x_lims_default, workspace_y_lims_default)
    sigmas = np.random.uniform(0.02, 0.15, size=(n_blocks, 2))
    sigmas = [[0.09814495, 0.11204944],
              [0.02267598, 0.14608828],
              [0.12821754, 0.04760408],
              [0.04363725, 0.04384259]]

    temp_explicit_belief = BlocksPositionsBelief(n_blocks, workspace_x_lims_default, workspace_y_lims_default,
                                                 mus, sigmas)
    initial_belief = explicit_belief_to_particles(temp_explicit_belief, num_particles=num_particles,
                                                  observable_steps_left=max_steps, observable_robot_position=tower_pos)

    initial_block_positions = sample_block_positions_from_dists(temp_explicit_belief.block_beliefs)
    initial_state = State(steps_left=max_steps, block_positions=initial_block_positions, robot_position=tower_pos)

    reward, planning_time = run_single_experiment(n_blocks, max_steps, num_sims, initial_belief, initial_state,
                                                  tower_pos,
                                                  max_planning_depth, experiment_params, show_progress=True)

    print(f"Total reward: {reward}")
    print(f"Average planning time per step: {planning_time} seconds")


def run_experiment_set(args):
    exp_id, n_blocks, max_steps, num_sims_list, max_planning_depths, initial_belief, initial_state, tower_pos, experiment_params, num_runs_per_experiment = args
    print(f"Running experiment {exp_id + 1}")
    exp_results = {ns: {'rewards': [], 'planning_times': []} for ns in num_sims_list}

    for num_sims, max_planning_depth in zip(num_sims_list, max_planning_depths):
        print(f"  Running with {num_sims} simulations")
        exp_rewards = []
        exp_planning_times = []

        for run in range(num_runs_per_experiment):
            reward, planning_time = run_single_experiment(n_blocks, max_steps, num_sims, initial_belief,
                                                          initial_state, tower_pos, max_planning_depth,
                                                          experiment_params)
            exp_rewards.append(reward)
            exp_planning_times.append(planning_time)

        exp_results[num_sims]['rewards'].extend(exp_rewards)
        exp_results[num_sims]['planning_times'].extend(exp_planning_times)

    return exp_results


@app.command()
def run_experiments(
        n_blocks: int = 4,
        max_steps: int = 20,
        stacking_reward: float = 1.0,
        sensing_cost_coeff: float = 0.05,
        stacking_cost_coeff: float = 0.05,
        finish_ahead_of_time_reward_coeff: float = 0.1,
        n_blocks_for_actions: int = 2,
        points_to_sample_for_each_block: int = 50,
        sensing_actions_to_sample_per_block: int = 2,
        sigmin: float = 0.02,
        sigmax: float = 0.15,
        num_experiments: int = 2,
        num_runs_per_experiment: int = 2,
        name: str = ""
):
    t = time.time()

    num_sims_list = [200, 1000, 2000]
    max_planning_depths = [4, 5, 5]
    tower_pos = goal_tower_position

    # set random seed:
    np.random.seed(42)

    experiment_params = {
        'stacking_reward': stacking_reward,
        'sensing_cost_coeff': sensing_cost_coeff,
        'stacking_cost_coeff': stacking_cost_coeff,
        'finish_ahead_of_time_reward_coeff': finish_ahead_of_time_reward_coeff,
        'n_blocks_for_actions': n_blocks_for_actions,
        'points_to_sample_for_each_block': points_to_sample_for_each_block,
        'sensing_actions_to_sample_per_block': sensing_actions_to_sample_per_block,
        'name': name
    }

    results = {ns: {'rewards': [], 'planning_times': []} for ns in num_sims_list}

    experiment_args = []
    for exp in range(num_experiments):
        # Sample initial belief
        mus = sample_block_positions_uniform(n_blocks, workspace_x_lims_default, workspace_y_lims_default)
        sigmas = np.random.uniform(sigmin, sigmax, size=(n_blocks, 2))
        initial_belief = BeliefModel(n_blocks, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

        # Sample initial state
        initial_block_positions = sample_block_positions_from_dists(initial_belief.block_beliefs)
        initial_state = State(steps_left=max_steps, block_positions=initial_block_positions, robot_position=tower_pos)

        experiment_args.append((exp, n_blocks, max_steps, num_sims_list, max_planning_depths, initial_belief,
                                initial_state, tower_pos, experiment_params, num_runs_per_experiment))

    # Use all available CPU cores
    num_processes = cpu_count() - 1

    with Pool(processes=num_processes) as pool:
        experiment_results = pool.map(run_experiment_set, experiment_args)

    for exp_results in experiment_results:
        for ns in num_sims_list:
            results[ns]['rewards'].extend(exp_results[ns]['rewards'])
            results[ns]['planning_times'].extend(exp_results[ns]['planning_times'])

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

    print(f"Total time: {time.time() - t:.2f} seconds")


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
