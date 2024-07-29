import numpy as np
import pomdp_py
from modeling.pomdp_problem.domain.state import *
from modeling.pomdp_problem.domain.action import *



""" No cost for now. To add cost need to add robot position to state and observation"""
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, stacking_reward=1,
                 finish_ahead_of_time_reward_coeff=0.1,
                 cost_coeff=0.1):
        self.stacking_reward = stacking_reward
        self.finish_ahead_of_time_reward_coeff = finish_ahead_of_time_reward_coeff
        self.cost_coeff = cost_coeff

    def sample(self, state, action, next_state):
        return self._reward_func(state, action, next_state)

    def _reward_func(self, state, action, next_state):
        reward = 0

        if isinstance(action, ActionAttemptStack):
            if next_state.last_stack_attempt_succeded:
                reward += self.stacking_reward

            if len(next_state.block_positions) == 0:
                reward += self.finish_ahead_of_time_reward_coeff * (state.steps_left - 1)

        return reward

