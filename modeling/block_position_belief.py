import numpy as np
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

    def update_from_point_sensing_observation(self, point_x, point_y, is_occupied):
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

    def update_from_image_detections_position_distribution(self, detection_mus, detection_sigmas):
        """
        This is practically p(s|o) for blocks, but not all blocks can be seen in the image.
        returns a list of mus and sigmas in the orders of their association with the blocks, where mu and sigma
        are -1 for blocks that were not associated with any detection.
        """
        assert len(detection_mus) == len(detection_sigmas)
        assert len(detection_mus) <= self.n_blocks, "More detections than blocks can cause a problem in" \
                                                    "the belief update. please remove detections with low confidence."

        # first associate each detection with a block
        # right now for simplicity each detection is associated with the block that has the highest likelihood
        detections_to_blocks = self._associate_detection_mus_with_blocks(detection_mus)

        for i in range(len(detections_to_blocks)):
            self.update_belief_block_from_detection(detections_to_blocks[i],
                                                    detection_mus[i],
                                                    detection_sigmas[i])

        mus_and_sigmas = []
        for i in range(self.n_blocks):
            if i in detections_to_blocks:
                mus = detection_mus[detections_to_blocks[i]]
                sigmas = detection_sigmas[detections_to_blocks[i]]
            else:
                mus = -1
                sigmas = -1
            mus_and_sigmas.append((mus, sigmas))

        return mus_and_sigmas

    def _associate_detection_mus_with_blocks(self, detection_mus):
        # Calculate likelihoods for each detection across all blocks
        per_detection_likelihoods = np.array([block_belief.pdf(detection_mus) for block_belief in self.block_beliefs])

        # Initially associate each detection with the block that has the highest likelihood
        detections_to_blocks = np.argmax(per_detection_likelihoods, axis=0)  # n_blocks x n_detections

        # Ensure unique association: resolve conflicts
        unique, counts = np.unique(detections_to_blocks, return_counts=True)
        while np.any(counts > 1):
            for block in unique[counts > 1]:
                # Indices of all detections currently assigned to this block
                conflicting_detections = np.where(detections_to_blocks == block)[0]
                best_conflicting_detection_id = np.argmax(
                    per_detection_likelihoods[block, conflicting_detections], axis=0)
                # Zero out likelihoods for all other blocks for this detection
                conflicting_detections_without_best = np.setdiff1d(
                    conflicting_detections, best_conflicting_detection_id)
                per_detection_likelihoods[block, conflicting_detections_without_best] = 0
                # now reassign the conflicting detections to the second-best block, which is now the best
                # one in the table
                detections_to_blocks[conflicting_detections_without_best] = np.argmax(
                    per_detection_likelihoods[:, conflicting_detections_without_best], axis=0)

            unique, counts = np.unique(detections_to_blocks, return_counts=True)

        return detections_to_blocks

    def update_belief_block_from_detection(self, block_id, mu_detection, sigma_detection):
        """
        Bayesian filter update where the initial belief is a gaussian defined by the expectation and std
        of the truncated masked gaussian distribution of its position (actually, this is not the real
        expectation and std, since we don't include masks in the computation)
        and the state given observation probability is also gaussian.
        """
        block_belief = self.block_beliefs[block_id]

        block_belief_mu_x = block_belief.dist_x.mean()
        block_belief_sigma_x = block_belief.dist_x.std()
        block_belief_mu_y = block_belief.dist_y.mean()
        block_belief_sigma_y = block_belief.dist_y.std()

        mu_detection_x, mu_detection_y = mu_detection
        sigma_detection_x, sigma_detection_y = sigma_detection

        # Calculate the updated parameters for x,
        updated_sigma_x_squared = 1 / (1 / block_belief_sigma_x ** 2 + 1 / sigma_detection_x ** 2)
        updated_mu_x = (block_belief_sigma_x ** 2 * mu_detection_x + sigma_detection_x ** 2 * block_belief_mu_x) / (
                    block_belief_sigma_x ** 2 + sigma_detection_x ** 2)
        updated_sigma_x = np.sqrt(updated_sigma_x_squared)

        updated_sigma_y_squared = 1 / (1 / block_belief_sigma_y ** 2 + 1 / sigma_detection_y ** 2)
        updated_mu_y = (block_belief_sigma_y ** 2 * mu_detection_y + sigma_detection_y ** 2 * block_belief_mu_y) / (
                    block_belief_sigma_y ** 2 + sigma_detection_y ** 2)
        updated_sigma_y = np.sqrt(updated_sigma_y_squared)

        block_belief.update_parameters(updated_mu_x, updated_sigma_x, updated_mu_y, updated_sigma_y)

    def add_empty_area(self, area_x_bounds, area_y_bounds):
        for block_belief in self.block_beliefs:
            block_belief.add_masked_area(np.array([area_x_bounds, area_y_bounds]))

