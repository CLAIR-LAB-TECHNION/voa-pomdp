from copy import deepcopy
import numpy as np
from pomdp_py import Particles

from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, ActionBase
from modeling.pomdp_problem.models.policy_model_v2 import PolicyModelV2
from modeling.pomdp_problem.models.reward_model import RewardModel
from modeling.pomdp_problem.models.observation_model import ObservationModel
from modeling.pomdp_problem.models.transition_model import TransitionModel
import pomdp_py


class AgentV2(pomdp_py.Agent):
    def __init__(self,
                 initial_blocks_position_belief: Particles,
                 tower_position,
                 successful_grasp_offset_x=0.015,
                 successful_grasp_offset_y=0.015,
                 n_sensing_actions_to_sample_per_block=4,
                 n_pickup_actions_to_sample_per_block=4,
                 action_points_std=0.05,
                 stacking_reward=1,
                 stacking_cost_coeff=0.0,
                 sensing_cost_coeff=0.0,
                 finish_ahead_of_time_reward_coeff=0.1):

        transition_model = TransitionModel(tower_position=tower_position,
                                           successful_grasp_offset_x=successful_grasp_offset_x,
                                           successful_grasp_offset_y=successful_grasp_offset_y)
        observation_model = ObservationModel(tower_position=tower_position)
        reward_model = RewardModel(stacking_reward=stacking_reward,
                                   sensing_cost_coeff=sensing_cost_coeff,
                                   stacking_cost_coeff=stacking_cost_coeff,
                                   finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff)

        policy_model = PolicyModelV2(n_sensing_actions_to_sample_per_block=n_sensing_actions_to_sample_per_block,
                                     n_pickup_actions_to_sample_per_block=n_pickup_actions_to_sample_per_block,
                                     action_points_std=action_points_std)

        super().__init__(init_belief=initial_blocks_position_belief,
                            transition_model=transition_model,
                            observation_model=observation_model,
                            reward_model=reward_model,
                            policy_model=policy_model,
                            name="block_stacking_agent_v2")
