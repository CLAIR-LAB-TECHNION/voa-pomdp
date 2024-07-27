import numpy as np
from modeling.belief.masked_gaussian_distribution import Masked2DTruncNorm

BlockPosDist = Masked2DTruncNorm


class BlocksPositionsBelief:
    sigma_for_uniform = 10  # this is std of 10 meters, almost uniform for our scales
    block_size = 0.04

    def __init__(self, n_blocks, ws_x_lims, ws_y_lims, init_mus=None, init_sigmas=None):
        self.n_blocks_orig = n_blocks
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

        self.n_blocks_on_table = n_blocks
        self.blocks_picked = np.zeros(n_blocks, dtype=bool)

    def update_from_point_sensing_observation(self, point_x, point_y, is_occupied, no_update_margin=0.005):
        """
        We update the area within block_size distance from the point, minus a margin of no_update_margin.
        the no update margin reflects that we have an error from the desired sensing point.
        """
        if not is_occupied:
            area_to_update = [[point_x - self.block_size + no_update_margin,
                               point_x + self.block_size - no_update_margin],
                              [point_y - self.block_size + no_update_margin,
                               point_y + self.block_size - no_update_margin]]
            # x and y is not occupied by a block. that means that there isn't a block withing
            # the block_size distance for each direction.
            area_to_update = [[point_x - self.block_size + no_update_margin,
                               point_x + self.block_size - no_update_margin],
                              [point_y - self.block_size + no_update_margin,
                               point_y + self.block_size - no_update_margin]]

            for block_belief in self.block_beliefs:
                block_belief.add_masked_area(area_to_update)
        else:
            area_to_update = [[point_x - self.block_size, point_x + self.block_size],
                              [point_y - self.block_size, point_y + self.block_size]]

            # x and y is occupied by a block. that means that there is a block withing
            # the block_size distance for each direction.
            # we associate it with the block that has the highest probability of being there
            blocks_probs = [b.pdf((point_x, point_y)) for b in self.block_beliefs]
            block_to_update_id = np.argmax(blocks_probs)
            block_to_update = self.block_beliefs[block_to_update_id]

            area_to_update = [[point_x - self.block_size - no_update_margin,
                               point_x + self.block_size + no_update_margin],
                              [point_y - self.block_size - no_update_margin,
                               point_y + self.block_size + no_update_margin]]
            block_to_update.add_new_bounds(area_to_update)

    def update_from_image_detections_position_distribution(self, detection_mus, detection_sigmas):
        """
        This is practically p(s|o) for blocks, but not all blocks can be seen in the image.
        returns a list of mus and sigmas in the orders of their association with the blocks, where mu and sigma
        are -1 for blocks that were not associated with any detection.
        """
        assert len(detection_mus) == len(detection_sigmas)
        assert len(detection_mus) <= self.n_blocks_on_table, "More detections than blocks can cause a problem in" \
                                                             "the belief update. please remove detections with low confidence."

        # first associate each detection with a block
        # right now for simplicity each detection is associated with the block that has the highest likelihood
        detections_to_blocks = self._associate_detection_mus_with_blocks(detection_mus)

        for i in range(len(detections_to_blocks)):
            self.update_belief_block_from_detection(detections_to_blocks[i],
                                                    detection_mus[i],
                                                    detection_sigmas[i])

        mus_and_sigmas = []
        for i in range(self.n_blocks_on_table):
            if i in detections_to_blocks:
                mus = detection_mus[detections_to_blocks[i]]
                sigmas = detection_sigmas[detections_to_blocks[i]]
            else:
                mus = -1
                sigmas = -1
            mus_and_sigmas.append((mus, sigmas))

        return mus_and_sigmas

    def update_from_successful_pick(self, pick_x, pick_y):
        """
        Update the belief after a successful pickup. The block is not on the workspace anymore
        """
        # remove the nearest block:
        block_to_remove_id = np.argmax([b.pdf((pick_x, pick_y)) for b in self.block_beliefs])

        self.block_beliefs.pop(block_to_remove_id)
        self.n_blocks_on_table -= 1
        self.blocks_picked[block_to_remove_id] = 1

        # there is nothing around this point anymore. Can use this information but carefully
        half_block_size = self.block_size / 2
        for block_belief in self.block_beliefs:
            block_belief.add_masked_area([[pick_x - half_block_size, pick_x + half_block_size],
                                          [pick_y - half_block_size, pick_y + half_block_size]])

    def update_from_history_of_sensing_and_pick_up(self,
                                                   positive_sensing_points,
                                                   negative_sensing_points,
                                                   successful_pickup_points,
                                                   no_update_margin=0.002):
        """
        This is almost equivalent to calls for update_from_point_sensing_observation for the positive
        and the negative sensing points and update_from_successful_pick for the successful picks.
        This is way more efficient since the masked areas overlap is happening once, this is usefull for
        rollouting during planning and it approximates the new belief after a history.
        The difference is that we don't consider order of observations, this can cause small changes
        in association of blocks with positive sensing or successful pickups, but most of the time it will
        generate the same result.
        """
        # first associate sensing with blocks
        per_block_positive_sensing_points = [[] for _ in range(self.n_blocks_on_table)]
        for point in positive_sensing_points:
            blocks_probs = [b.pdf((point[0], point[1])) for b in self.block_beliefs]
            block_to_update_id = np.argmax(blocks_probs)
            per_block_positive_sensing_points[block_to_update_id].append(point)

        # now adress negative sensing, generate a list of all the new masked areas
        masked_areas = []
        for point in negative_sensing_points:
            point_x, point_y = point
            area_to_mask = [[point_x - self.block_size + no_update_margin,
                             point_x + self.block_size - no_update_margin],
                            [point_y - self.block_size + no_update_margin,
                             point_y + self.block_size - no_update_margin]]
            masked_areas.append(area_to_mask)

        is_block_picked_up = [False] * self.n_blocks_on_table
        # now associate successful pickups points with blocks, and add the masked areas
        for point in successful_pickup_points:
            blocks_probs = [b.pdf((point[0], point[1])) for b in self.block_beliefs]
            block_to_update_id = np.argmax(blocks_probs)
            is_block_picked_up[block_to_update_id] = True
            # add small masked area:
            half_block_size = self.block_size / 2
            masked_areas.append([[point[0] - half_block_size, point[0] + half_block_size],
                                 [point[1] - half_block_size, point[1] + half_block_size]])

        # now update all the blocks.
        for i in range(self.n_blocks_on_table):
            # update just blocks that shouldn't be removed:
            if not is_block_picked_up[i]:
                new_bounds_list = []
                for pos_point in per_block_positive_sensing_points[i]:
                    point_x, point_y = pos_point
                    new_bounds_list.append([[point_x - self.block_size - no_update_margin,
                                             point_x + self.block_size + no_update_margin],
                                            [point_y - self.block_size - no_update_margin,
                                             point_y + self.block_size + no_update_margin]])
                self.block_beliefs[i].add_multiple_new_areas(masked_areas, new_bounds_list)

        # now remove the blocks that were picked up:
        for i in range(self.n_blocks_on_table):
            if is_block_picked_up[i]:
                self.block_beliefs.pop(i)
                self.n_blocks_on_table -= 1
                self.blocks_picked[i] = 1


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
