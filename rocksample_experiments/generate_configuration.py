import typer
import json
from typing import Dict, List, Tuple
import random
from rocksample_experiments.rocksample_problem import RockSampleProblem, RockType, State
from rocksample_experiments.help_actions import is_legal_rock_push

app = typer.Typer()


def generate_help_config(problem: RockSampleProblem, budget: int, max_rocks_to_push: int) -> dict:
    """
    Generate a single valid help configuration within budget by:
    1. Randomly selecting number of rocks to push (1 to max_rocks_to_push)
    2. Randomly dividing budget between selected rocks
    3. For each rock, generating dx, dy that sum to its allocated budget
    """
    while True:  # Main generation loop
        # Random number of rocks to push (1 to max_rocks_to_push)
        max_rocks = min(len(problem.rock_locs), max_rocks_to_push)
        num_rocks_to_push = random.randint(1, max_rocks)
        rocks_to_push = random.sample(list(problem.rock_locs.values()), num_rocks_to_push)
        rock_positions = {rock_id: pos for pos, rock_id in problem.rock_locs.items()}

        # Randomly divide budget between rocks
        rock_budgets = {}
        remaining_budget = budget

        for rock_id in rocks_to_push[:-1]:
            if remaining_budget <= 0:
                break
            rock_budgets[rock_id] = random.randint(1, remaining_budget)
            remaining_budget -= rock_budgets[rock_id]

        if remaining_budget > 0:
            rock_budgets[rocks_to_push[-1]] = remaining_budget

        help_config = {}
        total_used_budget = 0
        config_valid = True  # Flag to track if we need to retry entire configuration

        # For each rock, generate valid dx, dy considering grid boundaries
        for rock_id, rock_budget in rock_budgets.items():
            current_pos = rock_positions[rock_id]

            # Calculate maximum possible movements considering grid boundaries
            max_dx_neg = -current_pos[0]
            max_dx_pos = problem.n - 1 - current_pos[0]
            max_dy_neg = -current_pos[1]
            max_dy_pos = problem.n - 1 - current_pos[1]

            # Try to generate valid movement within boundaries
            valid_move_found = False
            attempts = 0
            while attempts < 10:
                dx_budget = random.randint(0, rock_budget)
                dy_budget = rock_budget - dx_budget

                dx = dx_budget * random.choice([-1, 1])
                dy = dy_budget * random.choice([-1, 1])

                if max_dx_neg <= dx <= max_dx_pos and max_dy_neg <= dy <= max_dy_pos:
                    help_config[rock_id] = (dx, dy)
                    total_used_budget += abs(dx) + abs(dy)
                    valid_move_found = True
                    break

                attempts += 1

            if not valid_move_found:
                config_valid = False  # Mark configuration as invalid
                break  # Break out of rock loop

        # Only check final conditions if all moves were valid
        if config_valid and total_used_budget == budget:
            if is_legal_rock_push(problem, help_config):
                return help_config


def generate_help_config_set(problem: RockSampleProblem, n_configs: int, budget: int, max_rocks_to_push: int) -> List[
    Dict]:
    """Generate n_configs unique valid help configurations"""
    configs = set()
    while len(configs) < n_configs:
        config = generate_help_config(problem, budget, max_rocks_to_push)
        # Convert dict to tuple for hashing
        config_tuple = tuple(sorted((k, v[0], v[1]) for k, v in config.items()))
        configs.add(config_tuple)

    # Convert back to list of dicts
    return [dict((k, (dx, dy)) for k, dx, dy in config) for config in configs]


def test_help_config_set(problem: RockSampleProblem, n_configs: int, budget: int, max_rocks_to_push: int):
    """
    Test help config set generation by validating:
    1. All configs are unique
    2. All configs use exactly the budget amount
    3. All configs are legal according to is_legal_rock_push
    """
    help_configs = generate_help_config_set(problem, n_configs, budget, max_rocks_to_push)

    # Test we got correct number of configs
    assert len(help_configs) == n_configs, f"Generated {len(help_configs)} configs instead of {n_configs}"

    # Test uniqueness
    config_tuples = [tuple(sorted((k, v[0], v[1]) for k, v in config.items())) for config in help_configs]
    assert len(set(config_tuples)) == n_configs, "Not all configurations are unique"

    for idx, config in enumerate(help_configs):
        # Test budget use
        total_budget_used = sum(abs(dx) + abs(dy) for dx, dy in config.values())
        assert total_budget_used == budget, f"Config {idx} uses {total_budget_used} budget instead of {budget}"

        # Test legality
        assert is_legal_rock_push(problem, config), f"Config {idx} is not legal"

        # Test number of rocks pushed doesn't exceed max
        assert len(
            config) <= max_rocks_to_push, f"Config {idx} pushes {len(config)} rocks, exceeding max of {max_rocks_to_push}"

        print(f"Config {idx}: {config}")

    print(f"All tests passed! Generated {n_configs} valid configurations.")


@app.command()
def test_generate_help_config_set():
    n, k = 15, 15
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
    problem = RockSampleProblem(
        n=n,
        k=k,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=None
    )

    # Test with different parameters
    test_help_config_set(problem, n_configs=100, budget=10, max_rocks_to_push=5)


@app.command()
def generate_experiment_configs(
        n: int = typer.Option(..., help="Grid size"),
        k: int = typer.Option(..., help="Number of rocks"),
        n_problem_instances: int = typer.Option(..., help="Number of problem instances"),
        n_actual_states: int = typer.Option(..., help="Number of states per instance"),
        n_help_configs: int = typer.Option(..., help="Number of help configurations per state"),
        push_help_budget: int = typer.Option(..., help="Budget for push help actions"),
        max_rocks_to_push: int = typer.Option(..., help="Maximum number of rocks to push in help actions"),
        output_file: str = typer.Option(..., help="Output JSON file path")
):
    """Generate experiment configurations and save to JSON file"""

    metadata = {
        "grid_size": n,
        "num_rocks": k,
        "num_problem_instances": n_problem_instances,
        "num_states_per_instance": n_actual_states,
        "num_help_configs": n_help_configs,
        "push_help_budget": push_help_budget,
        "max_rocks_to_push": max_rocks_to_push
    }

    experiments = []

    for env_idx in range(n_problem_instances):
        init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
        rover_position = init_state.position

        # Create problem instance for help config generation
        problem = RockSampleProblem(
            n=n,
            k=k,
            init_state=init_state,
            rock_locs=rock_locs,
            init_belief=None  # Not needed for help config generation
        )

        help_configs = generate_help_config_set(problem, n_help_configs, push_help_budget, max_rocks_to_push)

        # For each instance, generate different states (rock types)
        for state_idx in range(n_actual_states):
            # Generate new rock types
            rocktypes = tuple(RockType.random() for _ in range(k))

            # Add experiment with no help
            experiments.append({
                "env_instance_id": env_idx,
                "state_id": state_idx,
                "help_config_id": -1,
                "rover_position": list(rover_position),
                "rock_locations": {str(pos): rock_id for pos, rock_id in rock_locs.items()},
                "rock_types": list(rocktypes),
                "help_actions": {},
                "total_reward": None,
                "total_discounted_reward": None
            })

            # Add experiments with help configurations
            for help_idx, help_config in enumerate(help_configs):
                experiments.append({
                    "env_instance_id": env_idx,
                    "state_id": state_idx,
                    "help_config_id": help_idx,
                    "rover_position": list(rover_position),
                    "rock_locations": {str(pos): rock_id for pos, rock_id in rock_locs.items()},
                    "rock_types": list(rocktypes),
                    "help_actions": {str(k): list(v) for k, v in help_config.items()},
                    "total_reward": None,
                    "total_discounted_reward": None
                })

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump({"metadata": metadata, "experiments": experiments}, f, indent=2)


if __name__ == "__main__":
    app()