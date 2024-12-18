import random
import numpy as np
import pomdp_py
from rocksample_experiments.preferred_actions import CustomRSPolicyModel, RSActionPrior
from rocksample_experiments.rocksample_problem import RockSampleProblem, init_particles_belief
import typer
from typing import Optional
import os
import time


app = typer.Typer()


def run_pomcp_rocksample_instance(
        problem: RockSampleProblem,
        pomcp_params: dict,
        max_steps: int = 100,
        verbose: int = 0,
) -> tuple[float, float]:
    """
    Run a single RockSample experiment with given problem instance and POMCP parameters.

    Args:
        problem: Initialized RockSampleProblem instance
        pomcp_params: Dictionary containing POMCP configuration parameters
        max_steps: Maximum number of steps to run
        verbose: Verbosity level (0: none, 1: summary, 2+: detailed)

    Returns:
        tuple[float, float]: (total_reward, total_discounted_reward)
    """

    # Initialize POMCP planner
    pomcp = pomdp_py.POMCP(
        num_sims=pomcp_params.get('num_sims', 2000),
        max_depth=pomcp_params.get('max_depth', 20),
        discount_factor=pomcp_params.get('discount_factor', 0.95),
        exploration_const=pomcp_params.get('exploration_const', 5),
        action_prior=pomcp_params.get('action_prior', None),
        rollout_policy=pomcp_params.get('rollout_policy', problem.agent.policy_model),
        show_progress=verbose > 1
    )

    if verbose > 0:
        print("Initial state:")
        problem.print_state()

    # Run simulation
    total_reward = 0
    total_discounted_reward = 0
    gamma = 1.0
    discount = pomcp_params.get('discount_factor', 0.95)

    for step in range(max_steps):
        if verbose > 1:
            print(f"\nStep {step + 1}")

        # Plan and execute action
        action = pomcp.plan(problem.agent)
        reward = problem.env.state_transition(action, execute=True)

        # Get observation and update agent
        observation = problem.env.provide_observation(
            problem.agent.observation_model,
            action
        )
        problem.agent.update_history(action, observation)
        pomcp.update(problem.agent, action, observation)

        # Update rewards
        total_reward += reward
        total_discounted_reward += reward * gamma
        gamma *= discount

        if verbose > 1:
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Total Reward: {total_reward}")
            print(f"Total Discounted Reward: {total_discounted_reward}")
            problem.print_state()

        # Check if reached exit
        if problem.in_exit_area(problem.env.state.position):
            if verbose > 1:
                print("\nReached exit area! Terminating...")
            break

    if verbose > 0:
        print(f"\nExperiment completed:")
        print(f"Total Reward: {total_reward}")
        print(f"Total Discounted Reward: {total_discounted_reward}")

    return total_reward, total_discounted_reward


@app.command()
def run_rocksample_experiment(
        grid_size: int = typer.Option(11, help="Size of the grid (nxn)"),
        num_rocks: int = typer.Option(11, help="Number of rocks in the environment"),
        max_steps: int = typer.Option(100, help="Maximum number of steps to run"),
        n_sims: int = typer.Option(2000, help="Number of simulations for POMCP planner"),
        max_depth: int = typer.Option(20, help="Maximum depth for POMCP planner"),
        discount: float = typer.Option(0.95, help="Discount factor"),
        exploration_const: float = typer.Option(5.0, help="Exploration constant for POMCP"),
        half_efficiency_dist: float = typer.Option(20.0, help="Half efficiency distance for sensing"),
        num_particles: int = typer.Option(2000, help="Number of particles for belief representation"),
        use_preferred_actions: bool = typer.Option(True, help="Whether to use preferred actions in planning"),
        preferred_actions_v_init: float = typer.Option(10.0, help="Initial value for preferred actions"),
        preferred_actions_n_visits_init: int = typer.Option(10, help="Initial visit count for preferred actions"),
        seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
        verbose: int = typer.Option(1, help="Verbosity level (0: none, 1: summary, 2+: detailed)"),
        results_dir: str = typer.Option("", help="Directory to save results")
):
    """
    Run a single RockSample experiment with specified parameters.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Set up results directory
    if results_dir == "":
        results_dir = os.path.join(os.path.dirname(__file__), "experiments")

    datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(results_dir, datetime_stamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate initial state and belief
    init_state, rock_locs = RockSampleProblem.generate_instance(grid_size, num_rocks)

    init_belief = init_particles_belief(
        k=num_rocks,
        num_particles=num_particles,
        init_state=init_state,
        belief="uniform"
    )

    # Set up action prior if using preferred actions
    action_prior = None
    rollout_policy = None
    if use_preferred_actions:
        action_prior = RSActionPrior(
            grid_size,
            num_rocks,
            rock_locs,
            v_init=preferred_actions_v_init,
            n_visits_init=preferred_actions_n_visits_init
        )
        rollout_policy = CustomRSPolicyModel(grid_size, num_rocks, action_prior)

    # Create problem instance
    problem = RockSampleProblem(
        n=grid_size,
        k=num_rocks,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=init_belief,
        half_efficiency_dist=half_efficiency_dist
    )

    # Set up POMCP parameters
    pomcp_params = {
        'num_sims': n_sims,
        'max_depth': max_depth,
        'discount_factor': discount,
        'exploration_const': exploration_const,
        'action_prior': action_prior,
        'rollout_policy': rollout_policy if rollout_policy is not None else problem.agent.policy_model,
    }

    # Run experiment
    total_reward, total_discounted_reward = run_pomcp_rocksample_instance(
        problem=problem,
        pomcp_params=pomcp_params,
        max_steps=max_steps,
        verbose=verbose,
    )

    # Save results
    results = {
        'parameters': {
            'grid_size': grid_size,
            'num_rocks': num_rocks,
            'max_steps': max_steps,
            'n_sims': n_sims,
            'max_depth': max_depth,
            'discount': discount,
            'exploration_const': exploration_const,
            'half_efficiency_dist': half_efficiency_dist,
            'num_particles': num_particles,
            'use_preferred_actions': use_preferred_actions,
            'preferred_actions_v_init': preferred_actions_v_init,
            'preferred_actions_n_visits_init': preferred_actions_n_visits_init,
            'seed': seed
        },
        'results': {
            'total_reward': total_reward,
            'total_discounted_reward': total_discounted_reward
        }
    }

    import json
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    if verbose > 0:
        print(f"\nResults saved to: {experiment_dir}")
        print(f"Total Reward: {total_reward}")
        print(f"Total Discounted Reward: {total_discounted_reward}")


@app.command()
def main():
    pass

if __name__ == "__main__":
    app()