import copy
import random
from copy import deepcopy
import numpy as np
import pomdp_py
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, ActionBase
from modeling.belief.block_position_belief import BlocksPositionsBelief


class BeliefModel(BlocksPositionsBelief, pomdp_py.GenerativeDistribution):
    """
    just extend BlockPositionsBelief to implement GenerativeDistribution as well,
    so it can be used as belief by agent and policy
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # calls only BlocksPositionsBelief initializer

    def __new__(cls, *args, **kwargs):
        # Ensure proper creation of the instance using BlocksPositionsBelief's __new__
        instance = BlocksPositionsBelief.__new__(cls)
        return instance

    def __deepcopy__(self, memo):
        # Create a new instance without calling __init__
        new_instance = self.__class__.__new__(self.__class__)
        # Perform a deep copy of the instance's dictionary and update the new instance
        new_instance.__dict__ = copy.deepcopy(self.__dict__, memo)
        return new_instance

def history_to_belief(initial_belief: BeliefModel, history):
    # filter to sensing actions with positive sensing, sensing actions with negative sensing
    # and stack attempt actions with success

    sense_positive = []  # points where we sense there is a block
    sense_negative = []  # points where we sensed there isn't block
    stack_positive = []  # points where we stacked form

    for (action, observation) in history:
        if isinstance(action, ActionSense):
            if observation.is_occupied:
                sense_positive.append((observation.x, observation.y))
            else:
                sense_negative.append((observation.x, observation.y))
        elif isinstance(action, ActionAttemptStack):
            if observation.is_object_picked:
                stack_positive.append((action.x, action.y))

    new_belief = deepcopy(initial_belief)
    new_belief.update_from_history_of_sensing_and_pick_up(sense_positive, sense_negative, stack_positive)
    return new_belief


class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self,
                 initial_blocks_position_belief: BeliefModel,
                 points_to_sample_for_each_block=200,
                 sensing_actions_to_sample_per_block=2):
        self.initial_blocks_position_belief = initial_blocks_position_belief
        self.points_to_sample_for_each_block = points_to_sample_for_each_block
        self.sensing_actions_to_sample_per_block = sensing_actions_to_sample_per_block

    def get_all_actions(self, state, history, best_points=None):
        """
        this actually samples actions
        """
        if state.steps_left <= 0:
            return []

        actions_to_return: list[ActionBase] = []

        belief = history_to_belief(self.initial_blocks_position_belief, history)

        # first, sample 200 points from each block belief, and compute their pdfs
        per_block_points = []
        per_block_pdfs = []
        for block_dist in belief.block_beliefs:
            points = block_dist.sample_with_redundency(self.points_to_sample_for_each_block)
            per_block_points.append(points)
            per_block_pdfs.append(block_dist.pdf(points))

        # take k samples with highest pdf for each block:
        k = self.sensing_actions_to_sample_per_block
        per_block_best_points = []
        for pdfs, points in zip(per_block_pdfs, per_block_points):
            best_points_indices = np.argpartition(pdfs, -k)[-k:]
            best_points = points[best_points_indices]
            per_block_best_points.append(best_points)

        # choose a pickup action for each block. It should be picking up at approximately maximum likelihood position.
        for block_dist, best_points in zip(belief.block_beliefs, per_block_best_points):
            gaussian_center = (block_dist.mu_x, block_dist.mu_y)
            if block_dist.pdf(gaussian_center) != 0:
                # easy! this is the maximum likelihood!
                actions_to_return.append(ActionAttemptStack(gaussian_center[0], gaussian_center[1]))
            else:
                # find the best point
                best_point = best_points[-1]
                actions_to_return.append(ActionAttemptStack(best_point[0], best_point[1]))

        if state.steps_left == 1:
            # no use for sensing
            return actions_to_return

        # add sensing actions for all best points
        for best_points in per_block_best_points:
            actions_to_return += [ActionSense(point[0], point[1]) for point in best_points]

        return actions_to_return

    def rollout(self, state, history):
        if state.steps_left <= 0:
            return None
        return random.sample(self.get_all_actions(state, history), 1)[0]
