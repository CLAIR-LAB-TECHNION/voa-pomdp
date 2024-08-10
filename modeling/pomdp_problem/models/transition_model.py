from copy import deepcopy
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, DummyAction
import numpy as np
import pomdp_py


class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, tower_position, block_size=0.04, successful_grasp_offset_x=0.015,
                 successful_grasp_offset_y=0.015):
        self.tower_position = np.asarray(tower_position)
        self.block_size = block_size
        self.successful_grasp_offset_x = successful_grasp_offset_x
        self.successful_grasp_offset_y = successful_grasp_offset_y

    def sample(self, state, action):
        if isinstance(action, DummyAction):
            return state

        assert state.steps_left > 0

        if isinstance(action, ActionAttemptStack):
            return self.sample_stack_attempt(state, action)
        elif isinstance(action, ActionSense):
            return self.sample_sense(state, action)

    def sample_stack_attempt(self, state, action):
        next_state = deepcopy(state)
        next_state.steps_left -= 1
        # after pickup, the robot position is always at the tower position:
        next_state.robot_position = self.tower_position

        if len(next_state.block_positions) == 0:
            next_state.last_stack_attempt_succeded = False
            return next_state

        pick_pos = np.array([action.x, action.y])
        nearest_block = np.argmin(np.linalg.norm(state.block_positions - pick_pos, axis=1))
        nearest_block_pos = state.block_positions[nearest_block]

        # we assume deterministic here
        success = np.abs(nearest_block_pos[0] - pick_pos[0]) < self.successful_grasp_offset_x and \
                  np.abs(nearest_block_pos[1] - pick_pos[1]) < self.successful_grasp_offset_y

        if success:
            next_state.block_positions = np.delete(state.block_positions, nearest_block, axis=0)
            if len(next_state.block_positions) == 0:
                # And there's also a reward for that
                next_state.steps_left = 0

        next_state.last_stack_attempt_succeded = success



        return next_state

    def sample_sense(self, state, action):
        # sensing doesn't change the state, it only takes time and provides observation
        next_state = deepcopy(state)
        next_state.steps_left -= 1
        next_state.last_stack_attempt_succeded = None
        next_state.robot_position = np.array([action.x, action.y])
        return next_state

    def probability(self, next_state, state, action):
        raise NotImplementedError  # don't need this for sampling based methods?

    def argmax(self, state, action):
        raise NotImplementedError
