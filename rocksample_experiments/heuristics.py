from copy import deepcopy

import pomdp_py

from rocksample_experiments.help_actions import push_rocks
from rocksample_experiments.preferred_actions import RSActionPrior, CustomRSPolicyModel
from rocksample_experiments.rocksample_problem import RockSampleProblem


def plan_one_step_and_get_tree_values(problem: RockSampleProblem, pomcp_params: dict = None) -> dict:
    """
    Plan one step ahead and return the tree values for all actions.
    returns dict: {action: value}
    """
    if pomcp_params is None:
        pomcp_params = {}

    # don't modify the original problem
    problem = deepcopy(problem)

    pomcp = pomdp_py.POMCP(num_sims=pomcp_params.get('num_sims', 2000),
                           max_depth=pomcp_params.get('max_depth', 20),
                           discount_factor=pomcp_params.get('discount_factor', 0.95),
                           exploration_const=pomcp_params.get('exploration_const', 5),
                           action_prior=pomcp_params.get('action_prior', None),
                           rollout_policy=pomcp_params.get('rollout_policy', problem.agent.policy_model),
                           show_progress=False)

    action = pomcp.plan(problem.agent)
    tree = problem.agent.tree

    action_values = {action: child.value for action, child in tree.children.items()}
    return action_values


def first_step_planning_value(problem: RockSampleProblem,) -> float:
    action_prior = RSActionPrior(problem.n, problem.k, problem.rock_locs)
    pomcp_params = {"action_prior": action_prior,
                    "rollout_policy": CustomRSPolicyModel(problem.n, problem.k, actions_prior=action_prior),}
    action_values = plan_one_step_and_get_tree_values(problem, pomcp_params)
    action_values = action_values.values()
    return max(action_values)


def h_first_step_planning_value_diff(problem: RockSampleProblem, help_config, n_trials=1) -> float:
    problem_helped = deepcopy(problem)
    problem_helped, _ = push_rocks(problem_helped, help_config)

    vdiffs = []
    for _ in range(n_trials):
        vdiff = first_step_planning_value(problem_helped) - first_step_planning_value(problem)
        vdiffs.append(vdiff)
    return sum(vdiffs) / n_trials

def h_rollout_policy_value(problem: RockSampleProblem, help_config) -> float:
    pass


if __name__ == '__main__':
    from rocksample_experiments.utils import sample_problem_from_voa_row
    import pandas as pd

    row = pd.Series({
        'env_instance_id': 1,
        'help_config_id': 1,
        'help_actions': "{'10': [-1, -8], '7': [-5, -4], '1': [0, -8]}",
        'rover_position': "[0, 10]",
        'rock_locations': "{'(5, 4)': 0, '(2, 9)': 1, '(7, 4)': 2, '(9, 7)': 3, '(10, 9)': 4, '(8, 2)': 5, '(5, 3)': 6, '(6, 6)': 7, '(8, 0)': 8, '(10, 1)': 9, '(3, 10)': 10}",
        'empirical_voa': -9.262964,
        'empirical_voa_variance': 42.025261,
        'n_states': 40,
        'baseline_value': 14.601503,
        'std_error': 1.025003
    })

    problem = sample_problem_from_voa_row(row, 11)
    first_step_planning_value(problem)