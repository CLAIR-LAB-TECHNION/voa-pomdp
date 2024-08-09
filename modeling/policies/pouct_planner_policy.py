import logging

import numpy as np
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from modeling.pomdp_problem.models.policy_model import BeliefModel


class POUCTPolicy(AbastractPolicy):
    def __init__(self,
                 initial_belief: BlocksPositionsBelief,
                 max_steps: int,
                 tower_position: tuple[float, float],
                 max_planning_depth: int,
                 stacking_reward: float,
                 sensing_cost_coeff: float,
                 stacking_cost_coeff: float,
                 finish_ahead_of_time_reward_coeff: float,
                 n_blocks_for_actions: int,
                 points_to_sample_for_each_block: int,
                 sensing_actions_to_sample_per_block: int,
                 num_sims=2000,
                 show_progress=False):

        super().__init__()

        initial_belief_model = BeliefModel.from_block_positions_belief(initial_belief)
        self.agent = Agent(initial_blocks_position_belief=initial_belief_model,
                           max_steps=max_steps,
                           tower_position=tower_position,
                           stacking_reward=stacking_reward,
                           sensing_cost_coeff=sensing_cost_coeff,
                           stacking_cost_coeff=stacking_cost_coeff,
                           finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff,
                           n_blocks_for_actions=n_blocks_for_actions,
                           points_to_sample_for_each_block=points_to_sample_for_each_block,
                           sensing_actions_to_sample_per_block=sensing_actions_to_sample_per_block)

        self.planner = pomdp_py.POUCT(max_depth=max_planning_depth,
                                      num_sims=num_sims,
                                      discount_factor=1.0,
                                      rollout_policy=self.agent.policy_model,
                                      show_progress=show_progress)

        logging.info(f"POUCTPolicy initialized with the following parameters: "
                     f"max_steps={max_steps},"
                     f" tower_position={tower_position},"
                     f" max_planning_depth={max_planning_depth},"
                     f"stacking_reward={stacking_reward}"
                     f", sensing_cost_coeff={sensing_cost_coeff},"
                     f"stacking_cost_coeff={stacking_cost_coeff},"
                     f" finish_ahead_of_time_reward_coeff={finish_ahead_of_time_reward_coeff},"
                     f"n_blocks_for_actions={n_blocks_for_actions},"
                     f" points_to_sample_for_each_block={points_to_sample_for_each_block},"
                     f"sensing_actions_to_sample_per_block={sensing_actions_to_sample_per_block},"
                     f" num_sims={num_sims},"
                     f"show_progress={show_progress}")

    def sample_action(self,
                      positions_belief: BlocksPositionsBelief,
                      history: list[tuple[ActionBase, ObservationBase]]) -> ActionBase:

        belief_model = BeliefModel.from_block_positions_belief(positions_belief)
        self.agent._cur_belief = belief_model
        self.agent.set_belief(self.agent._cur_belief)  # only work this way due to weird cython stuff
        self.agent.update_history(history[-1][0], history[-1][1])
        self.planner.update(self.agent, history[-1][0], history[-1][1])  # updates tree but not belief

        for i in range(3):
            try:
                actual_action = self.planner.plan(self.agent)
                break
            except Exception as e:
                logging.warning(f"Failed to plan with error: {e}")
                if i == 2:
                    logging.error("Failed to plan 3 times, returning None")
                    return None
                continue

        logging.info(f"Planned action: {actual_action}")

        return actual_action
