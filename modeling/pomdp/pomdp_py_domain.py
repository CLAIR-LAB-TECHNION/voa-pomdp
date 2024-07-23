import pomdp_py
import numpy as np


class State(pomdp_py.State):
    def __init__(self, block_positions):
        self.block_positions = np.asarray(block_positions)
    def __eq__(self, other):
        return self.block_positions == other.block_positions

    def __hash__(self):
        return hash(tuple(self.block_positions))


class Observation(pomdp_py.Observation):
    def __init__(self, x, y, is_occupied):
        self.x = x
        self.y = y
        self.is_occupied = is_occupied

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.is_occupied == other.is_occupied

    def __hash__(self):
        return hash((self.x, self.y, self.is_occupied))


class Action(pomdp_py.Action):
    def __init__(self, type: str, x=None, y=None):
        assert type in ["sense_at", "stack_from"]
        self.type = type
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.type == other.type and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.type, self.x, self.y))