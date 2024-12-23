import numpy as np
import pomdp_py


class State(pomdp_py.State):
    def __init__(self, block_positions,  robot_position, steps_left, last_stack_attempt_succeded=None):
        self.block_positions = np.asarray(block_positions)
        self.robot_position = np.asarray(robot_position)
        self.steps_left = steps_left

        # this is meant to hold the information about success after stack attempt.
        # we need it because the observation depends only on next state on pomdp_py
        # and we need to know previous state. this is a small hack to overcome this.
        self.last_stack_attempt_succeded = last_stack_attempt_succeded

    def __eq__(self, other):
        return self.block_positions == other.block_positions and self.steps_left == other.steps_left \
            and self.robot_position == other.robot_position

    def __hash__(self):
        return hash((tuple(map(tuple, self.block_positions)), tuple(self.robot_position), self.steps_left,
                     self.last_stack_attempt_succeded))

    def __str__(self):
        return f"State(block_positions={self.block_positions},\n robot position={self.robot_position}, steps_left={self.steps_left})"
