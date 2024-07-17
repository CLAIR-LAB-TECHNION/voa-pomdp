import numpy as np
from scipy.stats import truncnorm
from collections import namedtuple
from modeling.masked_gaussian_distribution import Masked2DTruncNorm


BlockPosDist = Masked2DTruncNorm


class BlocksPositionsBelief:
    sigma_for_uniform = 10  # this is std of 10 meters, almost uniform for our scales
    block_size = 0.04

    def __init__(self, n_blocks, ws_x_lims, ws_y_lims, init_mus=None, init_sigmas=None):
        self.n_blocks = n_blocks
        self.ws_x_lims = ws_x_lims
        self.ws_y_lims = ws_y_lims

        if init_mus is None:
            init_mus = [np.mean(ws_x_lims), np.mean(ws_y_lims)]
        if init_sigmas is None:
            init_sigmas = [self.sigma_for_uniform, self.sigma_for_uniform]

        init_mus = np.array(init_mus)
        init_sigmas = np.array(init_sigmas)

        if init_mus.ndim == 1:
            init_mus = np.tile(init_mus, (n_blocks, 1))
        if init_sigmas.ndim == 1:
            init_sigmas = np.tile(init_sigmas, (n_blocks, 1))

        self.block_beliefs = [BlockPosDist(ws_x_lims, ws_y_lims, init_mus[i][0], init_sigmas[i][0],
                                           init_mus[i][1], init_sigmas[i][1])
                              for i in range(n_blocks)]

    def update_belief_from_point_sensing(self, point_x, point_y, is_occupied):
        if not is_occupied:
            # x and y is not occupied by a block. that means that there isn't a block withing
            # the block_size distance for each direction.
            for block_belief in self.block_beliefs:
                block_belief.add_masked_area([[point_x - self.block_size, point_x + self.block_size],
                                             [point_y - self.block_size, point_y + self.block_size]])

        else:
            # x and y is occupied by a block. that means that there is a block withing
            # the block_size distance for each direction.
            # we associate it with the block that has the highest probability of being there
            blocks_probs = [b.pdf((point_x, point_y)) for b in self.block_beliefs]
            block_to_update_id = np.argmax(blocks_probs)
            block_to_update = self.block_beliefs[block_to_update_id]
            block_to_update.add_new_bounds([[point_x - self.block_size, point_x + self.block_size],
                                            [point_y - self.block_size, point_y + self.block_size]])

    def add_empty_area(self, area_x_bounds, area_y_bounds):
        for block_belief in self.block_beliefs:
            block_belief.add_masked_area(np.array([area_x_bounds, area_y_bounds]))