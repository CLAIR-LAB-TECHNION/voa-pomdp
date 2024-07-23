import pomdp_py
import numpy as np


class State(pomdp_py.State):
    def __init__(self, block_positions, steps_left):
        self.block_positions = np.asarray(block_positions)
        self.steps_left = steps_left
    def __eq__(self, other):
        return self.block_positions == other.block_positions and self.steps_left == other.steps_left

    def __hash__(self):
        return hash((self.block_positions, self.steps_left))


class ObservationBase(pomdp_py.Observation):
    def __init__(self, steps_left):
        self.steps_left = steps_left

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


class ObservationSenseResult(ObservationBase):
    def __init__(self, x, y, is_occupied, steps_left):
        super(ObservationBase).__init__(steps_left)
        self.x = x
        self.y = y
        self.is_occupied = is_occupied

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.is_occupied == other.is_occupied\
            and self.steps_left == other.steps_left

    def __hash__(self):
        return hash((self.x, self.y, self.is_occupied, self.steps_left))

class ObservationStackAttemptResult(ObservationBase):
    def __init__(self, is_object_picked, steps_left):
        super(ObservationBase).__init__(steps_left)
        self.is_object_picked = is_object_picked

    def __eq__(self, other):
        return self.is_object_picked == other.is_object_picked and self.steps_left == other.steps_left

    def __hash__(self):
        return hash((self.is_object_picked, self.steps_left))


class ActionBase(pomdp_py.Action):
    def __init__(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        return hash((self.x, self.y))


class ActionPickUp(ActionBase):
    def __init__(self, x, y):
        super(ActionBase).__init__(x, y)

    def __eq__(self, other):
        return isinstance(other, ActionPickUp) and self.x == other.x and self.y == other.y


class ActionSense(ActionBase):
    def __init__(self, x, y):
        super(ActionBase).__init__(x, y)

    def __eq__(self, other):
        return isinstance(other, ActionSense) and self.x == other.x and self.y == other.y
