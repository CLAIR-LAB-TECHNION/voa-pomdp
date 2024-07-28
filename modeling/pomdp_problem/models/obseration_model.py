import numpy as np
import pomdp_py
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, block_size=0.04):
        self.block_size = block_size

    def sample(self, next_state, action):
        if action is None:
            return None

        if isinstance(action, ActionAttemptStack):
            # for simplicity, right now we assume that the robot can accurately sense if block is picked
            # We can make this probabilistic if we see this is not accurate enough
            return ObservationStackAttemptResult(next_state.last_stack_attempt_succeded, next_state.steps_left)
        elif isinstance(action, ActionSense):
            return self.sample_sense(next_state, action)

    def sample_sense(self, next_state, action):
        # check if there is block within block_size distance from the point both in x and y
        # if there is, it is occupied
        for block_pos in next_state.block_positions:
            if np.abs(block_pos[0] - action.x) < self.block_size and np.abs(block_pos[1] - action.y) < self.block_size:
                return ObservationSenseResult(action.x, action.y, True, next_state.steps_left)

        return ObservationSenseResult(action.x, action.y, False, next_state.steps_left)

    def probability(self, observation, next_state, action):
        raise NotImplementedError  # don't need this for sampling based methods?

    def argmax(self, next_state, action):
        raise NotImplementedError
