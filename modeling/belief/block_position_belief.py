import numpy as np
from modeling.belief.masked_gaussian_distribution import Masked2DTruncNorm, UnnormalizedMasked2DTruncNorm


BlockPosDist = Masked2DTruncNorm
UnnormalizedBlockPosDist = UnnormalizedMasked2DTruncNorm


class UnnormalizedBlocksPositionsBelief:
    sigma_for_uniform = 10
    block_size = 0.04

    def __init__(self, n_blocks,
                 ws_x_lims,
                 ws_y_lims,
                 init_mus=None,
                 init_sigmas=None,
                 successful_grasp_margin_x=0.015,
                 successful_grasp_margin_y=0.015,
                 build_block_beliefs=True):
        """
        build_block_beliefs - if False, additional intialization of block beliefs is required.
        the object is not usable until it's done. usefull for copying...
        """

        self.n_blocks_orig = n_blocks
        self.ws_x_lims = ws_x_lims
        self.ws_y_lims = ws_y_lims
        self.successful_grasp_margin_x = successful_grasp_margin_x
        self.successful_grasp_margin_y = successful_grasp_margin_y

        init_mus, init_sigmas = self._initialize_mus_sigmas(n_blocks, ws_x_lims, ws_y_lims, init_mus, init_sigmas)

        if build_block_beliefs:
            self.block_beliefs = self._create_block_beliefs(init_mus, init_sigmas)
            self.block_beliefs_original_position = [b for b in self.block_beliefs]
        else:
            self.block_beliefs = None
            self.block_beliefs_original_position = None

        self.n_blocks_on_table = n_blocks

    def _create_block_beliefs(self, init_mus, init_sigmas):
        return [UnnormalizedBlockPosDist(self.ws_x_lims, self.ws_y_lims,
                                         init_mus[i][0], init_sigmas[i][0],
                                         init_mus[i][1], init_sigmas[i][1])
                for i in range(self.n_blocks_orig)]

    def _initialize_mus_sigmas(self, n_blocks, ws_x_lims, ws_y_lims, init_mus, init_sigmas):
        if init_mus is None:
            init_mus = [np.mean(ws_x_lims), np.mean(ws_y_lims)]
        if init_sigmas is None:
            init_sigmas = [self.sigma_for_uniform, self.sigma_for_uniform]

        init_mus = np.asarray(init_mus)
        init_sigmas = np.asarray(init_sigmas)

        if init_mus.ndim == 1:
            init_mus = np.tile(init_mus, (n_blocks, 1))
        if init_sigmas.ndim == 1:
            init_sigmas = np.tile(init_sigmas, (n_blocks, 1))

        assert len(init_mus) == n_blocks
        assert len(init_sigmas) == n_blocks

        return init_mus, init_sigmas

    def update_from_point_sensing_observation(self, point_x, point_y, is_occupied, no_update_margin=0.01):
        if not is_occupied:
            self._update_negative_sensing(point_x, point_y, no_update_margin)
        else:
            self._update_positive_sensing(point_x, point_y, no_update_margin)

    def _update_negative_sensing(self, point_x, point_y, no_update_margin):
        area_to_update = self._calculate_sensing_area(point_x, point_y, no_update_margin)
        for block_belief in self.block_beliefs:
            block_belief.add_masked_area(area_to_update)

    def _update_positive_sensing(self, point_x, point_y, no_update_margin):
        blocks_probs = [b.pdf((point_x, point_y)) for b in self.block_beliefs]
        block_to_update_id = np.argmax(blocks_probs)
        block_to_update = self.block_beliefs[block_to_update_id]

        area_to_update = self._calculate_sensing_area(point_x, point_y, -no_update_margin)
        block_to_update.add_new_bounds(area_to_update)

    def _calculate_sensing_area(self, point_x, point_y, margin):
        return [[point_x - self.block_size + margin, point_x + self.block_size - margin],
                [point_y - self.block_size + margin, point_y + self.block_size - margin]]

    def update_from_pickup_attempt(self, pick_x, pick_y, observed_success):
        if observed_success:
            self._update_successful_pickup(pick_x, pick_y)
        else:
            self._update_failed_pickup(pick_x, pick_y)

    def _update_successful_pickup(self, pick_x, pick_y):
        block_to_remove_id = np.argmax([b.pdf((pick_x, pick_y)) for b in self.block_beliefs])
        removed = self.block_beliefs.pop(block_to_remove_id)
        for i in range(len(self.block_beliefs_original_position)):
            if self.block_beliefs_original_position[i] == removed:
                self.block_beliefs_original_position[i] = None
                break
        self.n_blocks_on_table -= 1

        half_block_size = self.block_size / 2
        self._add_empty_area([pick_x - half_block_size, pick_x + half_block_size],
                             [pick_y - half_block_size, pick_y + half_block_size])

    def _update_failed_pickup(self, pick_x, pick_y):
        self._add_empty_area([pick_x - self.successful_grasp_margin_x, pick_x + self.successful_grasp_margin_x],
                             [pick_y - self.successful_grasp_margin_y, pick_y + self.successful_grasp_margin_y])

    def _add_empty_area(self, area_x_bounds, area_y_bounds):
        for block_belief in self.block_beliefs:
            block_belief.add_masked_area(np.array([area_x_bounds, area_y_bounds]))

    def update_from_history_of_sensing_and_pick_up(self, positive_sensing_points, negative_sensing_points,
                                                   successful_pickup_points, non_successful_pickup_points,
                                                   no_update_margin=0.005):
        per_block_positive_sensing_points = self._associate_positive_sensing_points(positive_sensing_points)
        masked_areas = self._calculate_masked_areas(negative_sensing_points, successful_pickup_points,
                                                    non_successful_pickup_points, no_update_margin)

        self._update_blocks_from_history(per_block_positive_sensing_points, masked_areas,
                                         successful_pickup_points, no_update_margin)

    def _associate_positive_sensing_points(self, positive_sensing_points):
        # We use gaussian pdf here, without normalization, this is wrong but faster
        per_block_positive_sensing_points = [[] for _ in range(self.n_blocks_on_table)]
        for point in np.asarray(positive_sensing_points):
            # find potential blocks, where the point is not masked:
            potential_blocks = [i for i, b in enumerate(self.block_beliefs) if not b.are_points_masked(point)]
            potential_block_probs = [self.block_beliefs[i].gaussian_pdf(point.reshape(1, -1))
                                     for i in potential_blocks]
            block_to_update_id = potential_blocks[np.argmax(potential_block_probs)]
            per_block_positive_sensing_points[block_to_update_id].append(point)

        return per_block_positive_sensing_points

    def _calculate_masked_areas(self, negative_sensing_points, successful_pickup_points,
                                non_successful_pickup_points, no_update_margin):
        masked_areas = []
        for point in negative_sensing_points:
            masked_areas.append(self._calculate_sensing_area(point[0], point[1], no_update_margin))

        for point in successful_pickup_points:
            half_block_size = self.block_size / 2
            masked_areas.append([[point[0] - half_block_size, point[0] + half_block_size],
                                 [point[1] - half_block_size, point[1] + half_block_size]])

        for point in non_successful_pickup_points:
            masked_areas.append([[point[0] - self.successful_grasp_margin_x,
                                  point[0] + self.successful_grasp_margin_x],
                                 [point[1] - self.successful_grasp_margin_y,
                                  point[1] + self.successful_grasp_margin_y]])

        return masked_areas

    def _update_blocks_from_history(self, per_block_positive_sensing_points, masked_areas,
                                    successful_pickup_points, no_update_margin):
        # might be less accurate than with normal pdf, but faster
        is_block_picked_up = [False] * self.n_blocks_on_table
        for point in successful_pickup_points:
            blocks_probs = [b.gaussian_pdf(np.array([point]))
                            for b in self.block_beliefs]
            block_to_update_id = np.argmax(blocks_probs)
            is_block_picked_up[block_to_update_id] = True

        for i in range(self.n_blocks_on_table):
            if not is_block_picked_up[i]:
                new_bounds_list = []
                for pos_point in per_block_positive_sensing_points[i]:
                    new_bounds_list.append(self._calculate_sensing_area(pos_point[0], pos_point[1], -no_update_margin))
                self.block_beliefs[i].add_multiple_new_areas(masked_areas, new_bounds_list)

        self.block_beliefs = [b for i, b in enumerate(self.block_beliefs) if not is_block_picked_up[i]]
        self.n_blocks_on_table = len(self.block_beliefs)
        for i, is_picked in enumerate(is_block_picked_up):
            if is_picked:
                self.block_beliefs_original_position[i] = None


class BlocksPositionsBelief(UnnormalizedBlocksPositionsBelief):
    def _create_block_beliefs(self, init_mus, init_sigmas):
        return [BlockPosDist(self.ws_x_lims, self.ws_y_lims,
                             init_mus[i][0], init_sigmas[i][0],
                             init_mus[i][1], init_sigmas[i][1])
                for i in range(self.n_blocks_orig)]

    def _associate_positive_sensing_points(self, positive_sensing_points):
        # Don't use the fast gaussian pdf here, it doesn't consider normalizations...
        per_block_positive_sensing_points = [[] for _ in range(self.n_blocks_on_table)]
        for point in positive_sensing_points:
            # find potential blocks, where the point is not masked:
            potential_blocks = [i for i, b in enumerate(self.block_beliefs) if not b.are_points_masked(point)]
            potential_block_probs = [self.block_beliefs[i].pdf(point) for i in potential_blocks]
            block_to_update_id = potential_blocks[np.argmax(potential_block_probs)]
            per_block_positive_sensing_points[block_to_update_id].append(point)
        return per_block_positive_sensing_points

    def _update_blocks_from_history(self, per_block_positive_sensing_points, masked_areas,
                                    successful_pickup_points, no_update_margin):
        # The implementation in the base class uses the fast gaussian pdf which is not normalized
        # need to implement it with normal pdf
        raise NotImplementedError("This method is not implemented for the normalized version")

    def create_unnormalized(self):
        unnormalized = UnnormalizedBlocksPositionsBelief(self.n_blocks_orig,
                                                         self.ws_x_lims,
                                                         self.ws_y_lims,
                                                         successful_grasp_margin_x=self.successful_grasp_margin_x,
                                                         successful_grasp_margin_y=self.successful_grasp_margin_y,
                                                         build_block_beliefs=False)
        unnormalized.block_beliefs_original_position = [b.create_unnormalized()
                                                        for b in self.block_beliefs_original_position if b is not None]
        unnormalized.block_beliefs = [b for b in unnormalized.block_beliefs_original_position if b is not None]

        unnormalized.n_blocks_on_table = self.n_blocks_on_table
        return unnormalized

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

        mus = []
        sigmas = []
        for i in range(self.n_blocks_on_table):
            if i in detections_to_blocks:
                detected_mu = detection_mus[detections_to_blocks == i][0]
                mus.append(detected_mu)
                detected_sigma = detection_sigmas[detections_to_blocks == i][0]
                sigmas.append(detected_sigma)
            else:
                mus.append([-1, -1])
                sigmas.append([-1, -1])

        return mus, sigmas

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

    def _associate_detection_mus_with_blocks(self, detection_mus):
        # Calculate likelihoods for each detection across all blocks
        per_detection_likelihoods = np.array([block_belief.pdf(detection_mus) for block_belief in self.block_beliefs])

        # Initially associate each detection with the block that has the highest likelihood
        detections_to_blocks = np.argmax(per_detection_likelihoods, axis=0)

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
