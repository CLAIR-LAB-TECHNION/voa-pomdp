from typing import Iterable

import numpy as np
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from modeling.belief.masked_gaussian_distribution import Masked2DTruncNorm


class HandMadePolicy(AbastractPolicy):
    def __init__(self, square_size_for_cdf=0.01):
        super().__init__()
        self.square_size_for_cdf = square_size_for_cdf

    def sample_action(self,
                      positions_belief: BlocksPositionsBelief,
                      history: list[tuple[ActionBase, ObservationBase]]):
        """
            sample point with high likelihood for each block.
            if  for any of the block the probability that the block is inside a 1x1 cm
            square around that point is higher than 0.75, try pick up from that point.
            Otherwise, sense at one of the points that was sampled
        """
        per_block_max_likelihood_points = []
        per_block_max_cdf = []
        for block_belief in positions_belief.block_beliefs:
            points = block_belief.sample(100, max_retries=50)
            pdfs = block_belief.pdf(points)
            max_pdf_idx = np.argmax(pdfs)
            max_likelihood_point = points[max_pdf_idx]
            max_pdf = pdfs[max_pdf_idx]

            # compute cdf at

    def get_params(self) -> dict:
        return {"square_size_for_cdf": self.square_size_for_cdf}

    def reset(self):
        pass

