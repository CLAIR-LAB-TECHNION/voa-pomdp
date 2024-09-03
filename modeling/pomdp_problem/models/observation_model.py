import numpy as np
import pomdp_py
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult,\
    ObservationReachedTerminal
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, DummyAction


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, tower_position, block_size=0.04):
        self.block_size = block_size
        self.tower_position = tower_position

    def sample(self, next_state, action):
        if isinstance(action, DummyAction) or next_state.steps_left == 0 or\
                len(next_state.block_positions) == 0:
            return ObservationReachedTerminal()

        if isinstance(action, ActionAttemptStack):
            # for simplicity, right now we assume that the robot can accurately sense if block is picked
            # We can make this probabilistic if we see this is not accurate enough
            return ObservationStackAttemptResult(next_state.last_stack_attempt_succeded,
                                                 steps_left=next_state.steps_left,
                                                 robot_position=next_state.robot_position)
        elif isinstance(action, ActionSense):
            return self.sample_sense(next_state, action)

    def sample_sense(self, next_state, action):
        # check if there is block within block_size distance from the point both in x and y
        # if there is, it is occupied
        for block_pos in next_state.block_positions:
            if np.abs(block_pos[0] - action.x) < self.block_size/2 and np.abs(block_pos[1] - action.y) < self.block_size/2:
                return ObservationSenseResult(True,
                                              steps_left=next_state.steps_left,
                                              robot_position=next_state.robot_position)

        return ObservationSenseResult(False,
                                      steps_left=next_state.steps_left,
                                      robot_position=next_state.robot_position)

    def probability(self, observation, next_state, action):
        raise NotImplementedError  # don't need this for sampling based methods?

    def argmax(self, next_state, action):
        raise NotImplementedError
