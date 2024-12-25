from copy import deepcopy

import pomdp_py
from rocksample_experiments.preferred_actions import RSActionPrior, CustomRSPolicyModel
from rocksample_experiments.rocksample_problem import RockSampleProblem



def get_problem():
    pass


def plan_one_step_and_get_tree_values(problem: RockSampleProblem, pomcp_params: dict = None) -> dict:
    """
    Plan one step ahead and return the tree values for all actions.
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



def first_step_planning_value(problem: RockSampleProblem, ) -> float:
    pomcp_params = {"action_prior": RSActionPrior(...),
                    "rollout_policy": CustomRSPolicyModel(...),}