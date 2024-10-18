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
                 n_blocks_for_actions: int = 2,
                 points_to_sample_for_each_block: int = 50,
                 sensing_actions_to_sample_per_block: int = 2,
                 num_sims=2000,
                 show_progress=False):

        super().__init__()

        self.max_steps = max_steps
        self.tower_position = tower_position
        self.max_planning_depth = max_planning_depth
        self.stacking_reward = stacking_reward
        self.sensing_cost_coeff = sensing_cost_coeff
        self.stacking_cost_coeff = stacking_cost_coeff
        self.finish_ahead_of_time_reward_coeff = finish_ahead_of_time_reward_coeff
        self.n_blocks_for_actions = n_blocks_for_actions
        self.points_to_sample_for_each_block = points_to_sample_for_each_block
        self.sensing_actions_to_sample_per_block = sensing_actions_to_sample_per_block
        self.num_sims = num_sims
        self.show_progress = show_progress

        # will be set in reset:
        self.agent = None
        self.planner = None

        self.reset(initial_belief)

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
        self.agent.set_full_history(history)
        if len(history) > 0:
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

    def get_params(self) -> dict:
        return dict(max_steps=self.agent.max_steps,
                    tower_position=self.agent.tower_position,
                    max_planning_depth=self.planner.max_depth,
                    stacking_reward=self.agent.reward_model.stacking_reward,
                    sensing_cost_coeff=self.agent.reward_model.sensing_cost_coeff,
                    stacking_cost_coeff=self.agent.reward_model.stacking_cost_coeff,
                    finish_ahead_of_time_reward_coeff=self.agent.reward_model.finish_ahead_of_time_reward_coeff,
                    n_blocks_for_actions=self.agent.policy_model.n_blocks_for_actions,
                    points_to_sample_for_each_block=self.agent.policy_model.points_to_sample_for_each_block,
                    sensing_actions_to_sample_per_block=self.agent.policy_model.sensing_actions_to_sample_per_block,
                    num_sims=self.num_sims,)

    def reset(self, initial_belief: BlocksPositionsBelief):
        # It is the safest to just create new objects for each episode

        initial_belief_model = BeliefModel.from_block_positions_belief(initial_belief)
        self.agent = Agent(initial_blocks_position_belief=initial_belief_model,
                           max_steps=self.max_steps,
                           tower_position=self.tower_position,
                           stacking_reward=self.stacking_reward,
                           sensing_cost_coeff=self.sensing_cost_coeff,
                           stacking_cost_coeff=self.stacking_cost_coeff,
                           finish_ahead_of_time_reward_coeff=self.finish_ahead_of_time_reward_coeff,
                           n_blocks_for_actions=self.n_blocks_for_actions,
                           points_to_sample_for_each_block=self.points_to_sample_for_each_block,
                           sensing_actions_to_sample_per_block=self.sensing_actions_to_sample_per_block)

        self.planner = pomdp_py.POUCT(max_depth=self.max_planning_depth,
                                      num_sims=self.num_sims,
                                      discount_factor=1.0,
                                      rollout_policy=self.agent.policy_model,
                                      show_progress=self.show_progress)

        logging.info("POUCTPolicy reset")
