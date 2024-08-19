import copy
import random
from copy import deepcopy
import numpy as np
import pomdp_py
from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult, \
    ObservationReachedTerminal
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack, ActionBase, DummyAction
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.domain.state import State


import copy
from modeling.belief.block_position_belief import BlocksPositionsBelief
import pomdp_py

class BeliefModel(BlocksPositionsBelief, pomdp_py.GenerativeDistribution):
    """
    just extend BlockPositionsBelief to implement GenerativeDistribution as well,
    so it can be used as belief by agent and policy
    """
    def __init__(self, *args, **kwargs):
        BlocksPositionsBelief.__init__(self, *args, **kwargs)

    @classmethod
    def from_block_positions_belief(cls, belief: BlocksPositionsBelief):
        # deepcopy all attributes:
        new_belief = cls.__new__(cls)
        new_belief.__dict__ = copy.deepcopy(belief.__dict__)
        return new_belief

    def __new__(cls, *args, **kwargs):
        return BlocksPositionsBelief.__new__(cls)

    def __reduce__(self):
        # This method tells Python how to pickle the object
        return (BeliefModel._unpickle, (self.__dict__,))

    @staticmethod
    def _unpickle(state):
        # This method tells Python how to unpickle the object
        self = BeliefModel.__new__(BeliefModel)
        self.__dict__.update(state)
        return self

    def __deepcopy__(self, memo):
        new_instance = BeliefModel.__new__(BeliefModel)
        new_instance.__dict__ = copy.deepcopy(self.__dict__, memo)
        return new_instance


def history_to_unnormalized_belief(initial_belief: BeliefModel, history):
    # filter to sensing actions with positive sensing, sensing actions with negative sensing
    # and stack attempt actions with success

    sense_positive = []  # points where we sense there is a block
    sense_negative = []  # points where we sensed there isn't block
    stack_positive = []  # points where we stacked form
    stack_negative = []  # points where we tried to stack from but failed

    for (action, observation) in history:
        if isinstance(action, DummyAction) or isinstance(observation, ObservationReachedTerminal):
            break

        if isinstance(action, ActionSense):
            if observation.is_occupied:
                sense_positive.append((action.x, action.y))
            else:
                sense_negative.append((action.x, action.y))
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
                 n_blocks_for_actions=2,
                 points_to_sample_for_each_block=200,
                 sensing_actions_to_sample_per_block=2):
        self.initial_blocks_position_belief = initial_blocks_position_belief
        self.n_blocks_for_actions = n_blocks_for_actions
        self.points_to_sample_for_each_block = points_to_sample_for_each_block
        self.sensing_actions_to_sample_per_block = sensing_actions_to_sample_per_block

    def get_all_actions(self, state, history):
        """
        this actually samples actions
        """
        if (len(history) > 0 and (history[-1][1].steps_left <= 0)
                or len(state.block_positions) == 0):
            return [DummyAction()]

        belief = history_to_unnormalized_belief(self.initial_blocks_position_belief, history)
        if belief.block_beliefs == []:
            return [DummyAction()]

        focused_blocks = self.choose_focused_blocks(belief)

        actions_to_return: list[ActionBase] = []

        # first, sample 200 points from each block belief, and compute their pdfs
        per_block_points = []
        per_block_pdfs = []
        for block_dist in focused_blocks:
            points, pdfs = block_dist.very_fast_sample(self.points_to_sample_for_each_block,
                                                       return_pdfs=True)

            per_block_points.append(points)
            per_block_pdfs.append(pdfs)

        # take sample with max likelihood and k more random samples
        k = self.sensing_actions_to_sample_per_block - 1
        per_block_sense_points = []
        for pdfs, points in zip(per_block_pdfs, per_block_points):
            if len(points) == 0:
                # if no point sample after all attempts, belief maybe wrong due to wrong block accusation
                # of positive sensing. Nothing to do in that case...
                per_block_sense_points.append([])
            else:
                sense_points = [points[np.argmax(pdfs)]]
                sense_points += random.choices(points, k=k)
                per_block_sense_points.append(sense_points)

        # choose a pickup action for each block. It should be picking up at approximately maximum likelihood position.
        for block_dist, best_points in zip(focused_blocks, per_block_sense_points):
            gaussian_center = (block_dist.mu_x, block_dist.mu_y)
            if not block_dist.are_points_masked(gaussian_center) != 0:
                # easy! this is the maximum likelihood!
                actions_to_return.append(ActionAttemptStack(gaussian_center[0], gaussian_center[1]))
            elif len(best_points) > 0:
                # find the best point
                best_point = best_points[-1]
                actions_to_return.append(ActionAttemptStack(best_point[0], best_point[1]))

        if state.steps_left == 1:
            # no use for sensing
            return actions_to_return if actions_to_return else [DummyAction()]

        # add sensing actions for all best points
        for best_points in per_block_sense_points:
            actions_to_return += [ActionSense(point[0], point[1]) for point in best_points]

        if actions_to_return == []:
            return [DummyAction()]

        return actions_to_return

    def choose_focused_blocks(self, belief):
        """
        Choose self.n_blocks_for_actions blocks to sample actions for given the belief.
        Prioritize blocks with the most bounding areas. If on par,
        choose the blocks with the lowest variance and add score for
        nearby masked areas. Return their BlockDists.
        """
        block_scores = []
        for i, block_dist in enumerate(belief.block_beliefs):
            n_areas = len(block_dist.bounds_list)

            # Secondary score: inverse of variance (higher for lower variance)
            variance = block_dist.sigma_x + block_dist.sigma_y
            inverse_variance = 1 / (variance + 1e-10)

            # another score nearby masked areas
            nearby_masked_area_score = self._calculate_nearby_masked_area_score(block_dist)

            total_score = n_areas * 100 + inverse_variance + nearby_masked_area_score
            block_scores.append((i, total_score))

        sorted_blocks = sorted(block_scores, key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in sorted_blocks[:self.n_blocks_for_actions]]

        return [belief.block_beliefs[i] for i in selected_indices]

    def _calculate_nearby_masked_area_score(self, block_dist):
        score = 0
        center = (block_dist.mu_x, block_dist.mu_y)
        for masked_area in block_dist.masked_areas:
            # Calculate distance from block center to masked area center
            masked_center = ((masked_area[0][0] + masked_area[0][1]) / 2,
                             (masked_area[1][0] + masked_area[1][1]) / 2)
            distance = ((center[0] - masked_center[0]) ** 2 +
                        (center[1] - masked_center[1]) ** 2) ** 0.5

            # Add to score based on proximity (closer areas have higher score)
            score += 1 / (distance + 1e-10)

        return score

    def rollout(self, state, history) -> ActionBase:
        if state.steps_left <= 0 or len(state.block_positions) == 0:
            return DummyAction()
        # return random.choice(self.get_all_actions(state, history))

        # when rolling out, we will allways try a pick up since we should be quite deep in the tree
        # and have a really short rollout

        belief = history_to_unnormalized_belief(self.initial_blocks_position_belief, history)
        if belief.block_beliefs == []:
            return DummyAction()

        # find the block with the most bounding areas and the num of bounding areas:
        per_block_n_areas = [len(b.bounds_list) for b in belief.block_beliefs]
        max_n_areas = max(per_block_n_areas)
        if max_n_areas > 1:
            # try to pick up that block
            block_with_most_areas = np.argmax(per_block_n_areas)
            block_dist = belief.block_beliefs[block_with_most_areas]

            # sample 10 points and take the one with the highest pdf:
            points, pdfs = block_dist.very_fast_sample(20, max_retries=10, return_pdfs=True)
            if len(points) == 0:
                # no points to sample from, just return a random action
                return random.choice(self.get_all_actions(state, history))
            best_point = points[np.argmax(pdfs)]

            return ActionAttemptStack(best_point[0], best_point[1])

        # if no block position is known well enough, choose the one with lowest variance
        # and try to pick it up
        per_block_sigma = [np.sqrt(b.sigma_x + b.sigma_y) for b in belief.block_beliefs]
        block_with_least_variance = np.argmin(per_block_sigma)
        # sample 10 points and pick up from the one with highest pdf:
        block_dist = belief.block_beliefs[block_with_least_variance]
        points, pdfs = block_dist.very_fast_sample(10, return_pdfs=True)
        if len(points) == 0:
            # no points to sample from, just return a random action
            return random.choice(self.get_all_actions(state, history))
        best_point = points[np.argmax(pdfs)]

        return ActionAttemptStack(best_point[0], best_point[1])




