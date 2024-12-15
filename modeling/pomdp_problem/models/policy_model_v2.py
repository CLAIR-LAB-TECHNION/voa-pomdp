import pomdp_py
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult, \
    ObservationReachedTerminal
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, ActionBase, DummyAction
import numpy as np
import random

from modeling.pomdp_problem.domain.state import State


class PolicyModelV2(pomdp_py.RolloutPolicy):
    def __init__(self,
                 n_sensing_actions_to_sample_per_block=4,
                 n_pickup_actions_to_sample_per_block=4,
                 action_points_std=0.05,):
        self.n_sensing_actions_to_sample_per_block = n_sensing_actions_to_sample_per_block
        self.n_pickup_actions_to_sample_per_block = n_pickup_actions_to_sample_per_block
        self.action_points_std = action_points_std

    def get_all_actions(self, state: State, history):
        if state.steps_left == 0 or len(state.block_positions) == 0:
            return [DummyAction()]

        actions = []
        for block in state.block_positions:
            n_points = self.n_sensing_actions_to_sample_per_block + self.n_pickup_actions_to_sample_per_block
            points = np.random.normal(block, [self.action_points_std]*2, (n_points, 2))
            for point in points[:self.n_sensing_actions_to_sample_per_block]:
                actions.append(ActionSense(*point))
            for point in points[self.n_sensing_actions_to_sample_per_block:]:
                actions.append(ActionAttemptStack(*point))

        return actions

    def rollout(self, state: State, history):
        return random.choice(self.get_all_actions(state, history))

