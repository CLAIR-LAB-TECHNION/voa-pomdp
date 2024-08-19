import numpy as np
import pomdp_py
from modeling.pomdp_problem.domain.state import *
from modeling.pomdp_problem.domain.action import *



""" No cost for now. To add cost need to add robot position to state and observation"""
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, stacking_reward=1,
                 finish_ahead_of_time_reward_coeff=0.1,
                 sensing_cost_coeff=0.,
                 stacking_cost_coeff=0.,):
        self.stacking_reward = stacking_reward
        self.finish_ahead_of_time_reward_coeff = finish_ahead_of_time_reward_coeff

        self.sensing_cost_coeff = sensing_cost_coeff
        self.stacking_cost_coeff = stacking_cost_coeff

    def sample(self, state, action, next_state):
        return self._reward_func(state, action, next_state)

    def _reward_func(self, state, action, next_state):
        reward = 0

        if isinstance(action, ActionAttemptStack):
            pick_up_pos = np.array([action.x, action.y])
            pickup_movement_distance = np.linalg.norm(pick_up_pos - state.robot_position)
            put_down_movement_distance = np.linalg.norm(next_state.robot_position - pick_up_pos)
            reward -= self.stacking_cost_coeff * (pickup_movement_distance + put_down_movement_distance)

            if next_state.last_stack_attempt_succeded:
                reward += self.stacking_reward

            if len(next_state.block_positions) == 0:
                reward += self.finish_ahead_of_time_reward_coeff * (state.steps_left - 1)
        elif isinstance(action, ActionSense):
            reward -= self.sensing_cost_coeff * np.linalg.norm(next_state.robot_position - state.robot_position)

        return reward

