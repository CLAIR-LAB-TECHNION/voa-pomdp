import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import truncnorm
from collections import namedtuple

BlockPosDist = namedtuple("BlockPosDist", ["x_dist", "y_dist"])


class BlocksPositionsBelief:
    sigma_for_uniform = 10  # this is std of 10 meters, almost uniform for our scales

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

        self.block_beliefs = [BlockPosDist(self.truncated_gaussian(init_mus[i, 0], init_sigmas[i, 0], ws_x_lims),
                                           self.truncated_gaussian(init_mus[i, 1], init_sigmas[i, 1], ws_y_lims))
                              for i in range(n_blocks)]

    @staticmethod
    def truncated_gaussian(mu, sigma, limits):
        """
        Create a truncated Gaussian distribution from limits and mean/std.
        """
        a, b = (limits[0] - mu) / sigma, (limits[1] - mu) / sigma
        return truncnorm(a, b, loc=mu, scale=sigma)

    def set_block_belief_truncated_gaussian(self, block_id, x_mu, x_sigma, y_mu, y_sigma):
        self.block_beliefs[block_id] = BlockPosDist(
            self.truncated_gaussian(self.ws_x_lims, x_mu, x_sigma),
            self.truncated_gaussian(self.ws_y_lims, y_mu, y_sigma)
        )

    def plot_block_belief(self, block_id, actual_state=None):
        """
        plot the block position belief on a 2D plane by a block ID
        """
        self.plot_block_distribution(self.block_beliefs[block_id], actual_state=actual_state)




