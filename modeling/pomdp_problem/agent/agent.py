from copy import deepcopy

import numpy as np

from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, ActionBase
from modeling.pomdp_problem.models.policy_model import PolicyModel, BeliefModel
from modeling.pomdp_problem.models.reward_model import RewardModel
from modeling.pomdp_problem.models.obseration_model import ObservationModel
from modeling.pomdp_problem.models.transition_model import TransitionModel
import pomdp_py


class Agent(pomdp_py.Agent):
    def __init__(self,
                 initial_blocks_position_belief: BeliefModel,
                 max_steps,
                 tower_position,
                 successful_grasp_offset_x=0.015,
                 successful_grasp_offset_y=0.015,
                 n_blocks_for_actions=2,
                 points_to_sample_for_each_block=200,
                 sensing_actions_to_sample_per_block=2,
                 stacking_reward=1,
                 stacking_cost_coeff=0.0,
                 sensing_cost_coeff=0.0,
                 finish_ahead_of_time_reward_coeff=0.1):

        self.max_steps = max_steps
        self.tower_position = tower_position

        # cache management of samples from belief:
        self.need_to_generate_samples = True
        self.sampled_block_positions = None
        self.curr_unused_sampled_blockpos = None

        # initialization:
        transition_model = TransitionModel(tower_position=tower_position,
                                           successful_grasp_offset_x=successful_grasp_offset_x,
                                           successful_grasp_offset_y=successful_grasp_offset_y)
        observation_model = ObservationModel(tower_position=tower_position)
        reward_model = RewardModel(stacking_reward=stacking_reward,
                                   sensing_cost_coeff=sensing_cost_coeff,
                                   stacking_cost_coeff=stacking_cost_coeff,
                                   finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff)
        policy_model = PolicyModel(initial_blocks_position_belief=initial_blocks_position_belief,
                                   n_blocks_for_actions=n_blocks_for_actions,
                                   points_to_sample_for_each_block=points_to_sample_for_each_block,
                                   sensing_actions_to_sample_per_block=sensing_actions_to_sample_per_block)

        super().__init__(init_belief=initial_blocks_position_belief,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model,
                         policy_model=policy_model,
                         name="block_stacking_agent")

        # override what's done in the super class:
        self._cur_belief = deepcopy(initial_blocks_position_belief)

    def sample_belief(self):
        # this is override from base agent since we have our different belief model
        steps_left = self.max_steps - len(self.history)

        if self.need_to_generate_samples:
            # we don't have samples cached, so we need to generate them, generate 500 and save them
            block_positions_arrays = [block_dist.sample(500) for block_dist in self._cur_belief.block_beliefs]
            block_positions_arrays = np.array(block_positions_arrays)  # shape: (num_blocks, 500, 2)
            self.sampled_block_positions = block_positions_arrays.transpose(1, 0, 2)  # shape: (500, num_blocks, 2)

            self.curr_unused_sampled_blockpos = 0
            self.need_to_generate_samples = False

        self.curr_unused_sampled_blockpos += 1
        if self.curr_unused_sampled_blockpos == len(self.sampled_block_positions) - 1:
            self.need_to_generate_samples = True

        block_positions = self.sampled_block_positions[self.curr_unused_sampled_blockpos]

        robot_position = self.history[-1][1].robot_position if len(self.history) > 0 else self.tower_position
        return State(steps_left=steps_left,
                     block_positions=block_positions,
                     robot_position=robot_position,
                     last_stack_attempt_succeded=None)

    def update(self, actual_action, actual_observation):
        self.update_history(actual_action, actual_observation)
        self.update_belief(actual_action, actual_observation)

    def update_belief(self, actual_action, actual_observation):
        # reset cache:
        self.need_to_generate_samples = True
        self.sampled_block_positions = None
        self.curr_unused_sampled_blockpos = None

        if isinstance(actual_action, ActionSense):
            x, y = actual_action.x, actual_action.y
            is_occupied = actual_observation.is_occupied
            self._cur_belief.update_from_point_sensing_observation(x, y, is_occupied)
            self.set_belief(self._cur_belief)  # due to weird cpython stuff
        elif isinstance(actual_action, ActionAttemptStack):
            x, y = actual_action.x, actual_action.y
            self._cur_belief.update_from_pickup_attempt(x, y, actual_observation.is_object_picked)
            self.set_belief(self._cur_belief)  # due to weird cpython stuff
        else:
            raise NotImplementedError
