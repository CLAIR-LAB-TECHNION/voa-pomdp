import copy
import random
from copy import deepcopy
import numpy as np
import pomdp_py
from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, ActionBase, DummyAction
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.domain.state import State
from line_profiler_pycharm import profile


class BeliefModel(BlocksPositionsBelief, pomdp_py.GenerativeDistribution):
    """
    just extend BlockPositionsBelief to implement GenerativeDistribution as well,
    so it can be used as belief by agent and policy
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # calls only BlocksPositionsBelief initializer

    def __new__(cls, *args, **kwargs):
        # Ensure proper creation of the instance using BlocksPositionsBelief's __new__
        instance = UnnormalizedBlocksPositionsBelief.__new__(cls)
        return instance

    def __deepcopy__(self, memo):
        # Create a new instance without calling __init__
        new_instance = self.__class__.__new__(self.__class__)
        # Perform a deep copy of the instance's dictionary and update the new instance
        new_instance.__dict__ = copy.deepcopy(self.__dict__, memo)
        return new_instance

@profile
def history_to_unnormalized_belief(initial_belief: BeliefModel, history):
    # filter to sensing actions with positive sensing, sensing actions with negative sensing
    # and stack attempt actions with success

    sense_positive = []  # points where we sense there is a block
    sense_negative = []  # points where we sensed there isn't block
    stack_positive = []  # points where we stacked form
    stack_negative = []  # points where we tried to stack from but failed

    for (action, observation) in history:
        if isinstance(action, ActionSense):
            if observation.is_occupied:
                sense_positive.append((observation.x, observation.y))
            else:
                sense_negative.append((observation.x, observation.y))
        elif isinstance(action, ActionAttemptStack):
            if observation.is_object_picked:
                stack_positive.append((action.x, action.y))
            else:
                stack_negative.append((action.x, action.y))

    new_belief = initial_belief.create_unnormalized()
    new_belief.update_from_history_of_sensing_and_pick_up(sense_positive,
                                                          sense_negative,
                                                          stack_positive,
                                                          stack_negative)
    return new_belief

class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self,
                 initial_blocks_position_belief: BeliefModel,
                 points_to_sample_for_each_block=200,
                 sensing_actions_to_sample_per_block=2):
        self.initial_blocks_position_belief = initial_blocks_position_belief
        self.points_to_sample_for_each_block = points_to_sample_for_each_block
        self.sensing_actions_to_sample_per_block = sensing_actions_to_sample_per_block

    @profile
    def get_all_actions(self, state, history):
        """
        this actually samples actions
        """
        if state.steps_left <= 0:
            return [DummyAction()]

        actions_to_return: list[ActionBase] = []

        belief = history_to_unnormalized_belief(self.initial_blocks_position_belief, history)

        # first, sample 200 points from each block belief, and compute their pdfs
        per_block_points = []
        per_block_pdfs = []
        for block_dist in belief.block_beliefs:
            points, pdfs = block_dist.very_fast_sample(self.points_to_sample_for_each_block,
                                                       return_pdfs=True)
            per_block_points.append(points)
            per_block_pdfs.append(pdfs)

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
            if not block_dist.are_points_masked(gaussian_center) != 0:
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

    @profile
    def rollout(self, state, history) -> ActionBase:
        if state.steps_left <= 0:
            return DummyAction()
        # return random.choice(self.get_all_actions(state, history))

        # when rolling out, we will allways try a pick up since we should be quite deep in the tree
        # and have a really short rollout

        belief = history_to_unnormalized_belief(self.initial_blocks_position_belief, history)

        # find the block with the most bounding areas and the num of bounding areas:
        per_block_n_areas = [len(b.bounds_list) for b in belief.block_beliefs]
        max_n_areas = max(per_block_n_areas)
        if max_n_areas > 1:
            # try to pick up that block
            block_with_most_areas = np.argmax(per_block_n_areas)
            block_dist = belief.block_beliefs[block_with_most_areas]

            # sample 10 points and take the one with the highest pdf:
            points, pdfs = block_dist.very_fast_sample(20, max_retries=20, return_pdfs=True)
            best_point = points[np.argmax(pdfs)]

            return ActionAttemptStack(best_point[0], best_point[1])

        # if no block position is known well enough, choose the one with lowest variance
        # and try to pick it up
        per_block_sigma = [np.sqrt(b.sigma_x + b.sigma_y) for b in belief.block_beliefs]
        block_with_least_variance = np.argmin(per_block_sigma)
        # sample 10 points and pick up from the one with highest pdf:
        block_dist = belief.block_beliefs[block_with_least_variance]
        points, pdfs = block_dist.very_fast_sample(10, return_pdfs=True)
        best_point = points[np.argmax(pdfs)]

        return ActionAttemptStack(best_point[0], best_point[1])




