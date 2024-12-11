import logging
from copy import deepcopy

import numpy as np
from modeling.pomdp_problem.agent.agent_v2 import AgentV2
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from modeling.policies.abstract_policy import AbstractPolicy
from modeling.pomdp_problem.models.policy_model_v2 import PolicyModelV2
from pomdp_py import Particles


class POMCPPolicy(AbstractPolicy):
    def __init__(self,
                    initial_belief: Particles,
                    max_steps: int,
                    tower_position: tuple[float, float],
                    max_planning_depth: int,
                    num_sims=2000,
                    stacking_reward: float = 1,
                    stacking_cost_coeff: float = 0.0,
                    sensing_cost_coeff: float = 0.0,
                    finish_ahead_of_time_reward_coeff: float = 0.1,
                    n_sensing_actions_to_sample_per_block: int = 4,
                    n_pickup_actions_to_sample_per_block: int = 4,
                    action_points_std: float = 0.05,
                    show_progress=False):

            super().__init__()

            self.max_steps = max_steps
            self.tower_position = tower_position
            self.max_planning_depth = max_planning_depth
            self.num_sims = num_sims
            self.n_sensing_actions_to_sample_per_block = n_sensing_actions_to_sample_per_block
            self.n_pickup_actions_to_sample_per_block = n_pickup_actions_to_sample_per_block
            self.action_points_std = action_points_std
            self.stacking_reward = stacking_reward
            self.stacking_cost_coeff = stacking_cost_coeff
            self.sensing_cost_coeff = sensing_cost_coeff
            self.finish_ahead_of_time_reward_coeff = finish_ahead_of_time_reward_coeff
            self.show_progress = show_progress

            self.agent: AgentV2 = None
            self.planner: pomdp_py.POMCP = None

            self.reset(initial_belief)

            logging.info(f"POMCPPolicy initialized with the following parameters: "
                        f"max_steps={max_steps},"
                        f" tower_position={tower_position},"
                        f" n_sensing_actions_to_sample_per_block={n_sensing_actions_to_sample_per_block},"
                        f" n_pickup_actions_to_sample_per_block={n_pickup_actions_to_sample_per_block},"
                        f" action_points_std={action_points_std},"
                        f" stacking_reward={stacking_reward},"
                        f" stacking_cost_coeff={stacking_cost_coeff},"
                        f" sensing_cost_coeff={sensing_cost_coeff},"
                        f" finish_ahead_of_time_reward_coeff={finish_ahead_of_time_reward_coeff}")

    def sample_action(self, state, history):
        for i in range(3):
            try:
                actual_action = self.planner.plan(self.agent)
                break
            except Exception as e:
                logging.error(f"Error in planning: {e}")
                print(f"Error in planning: {e}")
                self.reset(self.agent._cur_belief)
                if i == 2:
                    logging.error("Failed to plan after 3 attempts")
                    print("Failed to plan after 3 attempts")
                    return None

        logging.info(f"Planned action: {actual_action}")
        return actual_action

    def get_belief(self):
        return deepcopy(self.agent.belief)

    def reset(self, initial_belief: Particles):
        # It is the safest to just create new objects for each episode

        self.agent = AgentV2(initial_blocks_position_belief=initial_belief,
                             tower_position=self.tower_position,
                             action_points_std=self.action_points_std,
                             n_sensing_actions_to_sample_per_block=self.n_sensing_actions_to_sample_per_block,
                             n_pickup_actions_to_sample_per_block=self.n_pickup_actions_to_sample_per_block,
                             stacking_reward=self.stacking_reward,
                             stacking_cost_coeff=self.stacking_cost_coeff,
                             sensing_cost_coeff=self.sensing_cost_coeff,
                             finish_ahead_of_time_reward_coeff=self.finish_ahead_of_time_reward_coeff)

        self.planner = pomdp_py.POMCP(max_depth=self.max_planning_depth,
                                      num_sims=self.num_sims,
                                      discount_factor=1.,
                                      # exploration_const=,
                                      # action_prior=None,
                                      rollout_policy=self.agent.policy_model,
                                      show_progress=self.show_progress,)

        logging.info("POMCPPolicy reset")


    def get_params(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "tower_position": self.tower_position,
            "n_sensing_actions_to_sample_per_block": self.n_sensing_actions_to_sample_per_block,
            "n_pickup_actions_to_sample_per_block": self.n_pickup_actions_to_sample_per_block,
            "action_points_std": self.action_points_std,
            "stacking_reward": self.stacking_reward,
            "stacking_cost_coeff": self.stacking_cost_coeff,
            "sensing_cost_coeff": self.sensing_cost_coeff,
            "finish_ahead_of_time_reward_coeff": self.finish_ahead_of_time_reward_coeff
        }