from typing import Iterable
import time
import numpy as np
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from modeling.belief.masked_gaussian_distribution import Masked2DTruncNorm


class HandMadePolicy(AbastractPolicy):
    def __init__(self,
                 square_half_size_for_cdf=0.015,
                 confidence_for_stack=0.6,
                 points_to_sample=30):
        super().__init__()
        self.square_half_size_for_cdf = square_half_size_for_cdf
        self.confidence_for_stack = confidence_for_stack
        self.points_to_sample = points_to_sample

    def sample_action(self,
                      positions_belief: BlocksPositionsBelief,
                      history: list[tuple[ActionBase, ObservationBase]]):
        """
        sample point with high likelihood for each block.
        if  for any of the block the probability that the block is inside a 2x2 cm
        square around that point is higher than 0.7, try pick up from that point.
        Otherwise, sense at one of the points that was sampled
        """
        per_block_max_cdf_points = []
        per_block_max_cdf = []
        for block_belief in positions_belief.block_beliefs:
            points = block_belief.sample(self.points_to_sample, max_retries=50)

            max_cdf_point = None
            max_cdf = 0
            # find the point with max cdf in a square around
            for p in points:
                minx = p[0] - self.square_half_size_for_cdf
                maxx = p[0] + self.square_half_size_for_cdf
                miny = p[1] - self.square_half_size_for_cdf
                maxy = p[1] + self.square_half_size_for_cdf
                cdf = block_belief.cdf(minx, maxx, miny, maxy)
                if cdf > max_cdf:
                    max_cdf = cdf
                    max_cdf_point = p

            per_block_max_cdf_points.append(max_cdf_point)
            per_block_max_cdf.append(max_cdf)

        # sort cdfs and points accordingly:
        sorted_indices = np.argsort(per_block_max_cdf)[::-1]


        # if there is block in a small square with high probability, try to pick up from that point
        # (or if that's the last step anyway)
        if per_block_max_cdf[sorted_indices[0]] > self.confidence_for_stack or \
                (len(history) > 0 and history[-1][1].steps_left == 1):
            return ActionAttemptStack(*per_block_max_cdf_points[sorted_indices[0]])

        # otherwise sense at sampled point with the max pdf, but first make sure we haven't:
        # already sensed too close to that point
        point_to_sense = per_block_max_cdf_points[sorted_indices[0]]
        all_sensed_points = [(action.x, action.y) for action, _ in history if isinstance(action, ActionSense)]
        if len(all_sensed_points) == 0:
            return ActionSense(*point_to_sense)

        all_sensed_points = np.array(all_sensed_points)
        distances = np.linalg.norm(all_sensed_points - point_to_sense, axis=1)
        while np.min(distances) < 0.003:
            point_to_sense += np.random.uniform(-0.005, 0.005, 2)
            distances = np.linalg.norm(all_sensed_points - point_to_sense, axis=1)

        return ActionSense(*point_to_sense)


    def get_params(self) -> dict:
        return {"square_half_size_for_cdf": self.square_half_size_for_cdf,
                "confidence_for_stack": self.confidence_for_stack,
                "points_to_sample": self.points_to_sample}

    def reset(self, initial_belief):
        pass

