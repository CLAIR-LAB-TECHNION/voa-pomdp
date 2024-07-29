import numpy as np
from scipy.stats import truncnorm
from modeling.belief.rectangles_overlap_resolution import resolve_overlaps, resolve_overlaps_only_for_new


def get_truncnorm_distribution(bounds, mu, sigma):
    a, b = (bounds[0] - mu) / sigma, (bounds[1] - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)


class Masked2DTruncNorm:
    def __init__(self, bounds_x, bounds_y, mu_x, sigma_x, mu_y, sigma_y):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y

        self.masked_areas = []  # [[x,x] [y,y]] of areas that the distribution is zero

        self.dist_x = get_truncnorm_distribution(bounds_x, mu_x, sigma_x)
        self.dist_y = get_truncnorm_distribution(bounds_y, mu_y, sigma_y)

        self.normalization_constant = None

    def add_masked_area(self, mask_bounds):
        # Add new mask and merge overlapping masks
        self.masked_areas.append(mask_bounds)
        self.masked_areas = resolve_overlaps(self.masked_areas)
        self.normalization_constant = self._calculate_normalization_constant()

    def add_new_bounds(self, bounds):
        # add bounds that the blocks must be in, or, the distribution will be zero out of these bounds
        # just create masked areas out of these bounds, instead of maintaining those bounds
        new_bounds_x, new_bounds_y = bounds

        new_masked_areas = [
            [[self.bounds_x[0], new_bounds_x[0]], self.bounds_y],
            [[new_bounds_x[1], self.bounds_x[1]], self.bounds_y],
            [new_bounds_x, [self.bounds_y[0], new_bounds_y[0]]],
            [new_bounds_x, [new_bounds_y[1], self.bounds_y[1]]],
        ]

        # since those areas are usually large, and going to contain other masked areas if there are so,
        # it's better to add them at the beginning of the list for the overlap resolution algorithm
        self.masked_areas = new_masked_areas + self.masked_areas
        self.masked_areas = resolve_overlaps(self.masked_areas)
        self.normalization_constant = self._calculate_normalization_constant()

    def add_multiple_new_areas(self, masked_areas, new_bounds):
        """
        equivalent to calling add_masked_area and add_new_bounds for all areas and abounds, but this one is
        way more efficient since overlaps is resolved only once for all new areas
        """
        # first add all the new bounds at the beginning:
        for bound in new_bounds:
            new_bounds_x, new_bounds_y = bound
            masked_areas_from_bounds = [
                [[self.bounds_x[0], new_bounds_x[0]], self.bounds_y],
                [[new_bounds_x[1], self.bounds_x[1]], self.bounds_y],
                [new_bounds_x, [self.bounds_y[0], new_bounds_y[0]]],
                [new_bounds_x, [new_bounds_y[1], self.bounds_y[1]]], ]
            self.masked_areas = masked_areas_from_bounds + self.masked_areas

        # now add the new masked areas at the end:
        self.masked_areas = self.masked_areas + masked_areas

        # now resolve overlaps for all:
        if len(self.masked_areas) > 0:
            self.masked_areas = resolve_overlaps(self.masked_areas)
            self.normalization_constant = self._calculate_normalization_constant()

    def _calculate_normalization_constant(self):
        """Calculates the normalization constant for the joint PDF."""
        full_probability = 1  # Assume full probability mass covers the entire normalized range
        for mask in self.masked_areas:
            masked_probability = self._calculate_masked_probability(mask, self.bounds_x, self.mu_x, self.sigma_x,
                                                                    self.bounds_y, self.mu_y, self.sigma_y)
            full_probability -= masked_probability

        if full_probability == 0:
            full_probability = 1e-9
        return full_probability

    def _calculate_probability_mass(self, mask_dim, bounds, mu, sigma):
        """Calculates the probability mass under the truncated normal curve for a given dimension."""
        start, end = mask_dim
        return truncnorm.cdf(end, (bounds[0] - mu) / sigma, (bounds[1] - mu) / sigma, loc=mu, scale=sigma) \
            - truncnorm.cdf(start, (bounds[0] - mu) / sigma, (bounds[1] - mu) / sigma, loc=mu, scale=sigma)

    def _calculate_masked_probability(self, mask, bounds_x, mu_x, sigma_x, bounds_y, mu_y, sigma_y):
        """Calculates the probability mass of a masked region for the given dimensions."""
        # Calculate the probability mass in the x dimension
        prob_mass_x = self._calculate_probability_mass(mask[0], bounds_x, mu_x, sigma_x)

        # Calculate the probability mass in the y dimension
        prob_mass_y = self._calculate_probability_mass(mask[1], bounds_y, mu_y, sigma_y)

        # Return the product of the probability masses in both dimensions
        return prob_mass_x * prob_mass_y

    def update_parameters(self, mu_x, sigma_x, mu_y, sigma_y):
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y

        self.dist_x = get_truncnorm_distribution(self.bounds_x, mu_x, sigma_x)
        self.dist_y = get_truncnorm_distribution(self.bounds_y, mu_y, sigma_y)

        self.normalization_constant = self._calculate_normalization_constant()

    def pdf(self, points):
        # Ensure normalization constant is calculated
        if self.normalization_constant is None:
            self.normalization_constant = self._calculate_normalization_constant()

        # Extract x and y coordinates
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        x = points[:, 0]
        y = points[:, 1]

        norm_const = self.normalization_constant if self.normalization_constant != 0 else 1e-8
        # Vectorized check for masked areas
        result = (self.dist_x.pdf(x) * self.dist_y.pdf(y)) / self.normalization_constant
        for mask in self.masked_areas:
            mask_condition = (mask[0][0] <= x) & (x <= mask[0][1]) & (mask[1][0] <= y) & (y <= mask[1][1])
            result[mask_condition] = 0
        return result

    def sample(self, n_samples=1):
        # Ensure normalization constant is calculated
        if self.normalization_constant is None:
            self.normalization_constant = self._calculate_normalization_constant()

        samples_x = self.dist_x.rvs(n_samples)
        samples_y = self.dist_y.rvs(n_samples)
        points = np.stack([samples_x, samples_y], axis=1)

        # make sure none of the points are in masked areas:
        while True:
            invalid_samples = np.where(self.pdf(points) == 0)[0]
            if len(invalid_samples) == 0:
                break

            new_samples_x = self.dist_x.rvs(len(invalid_samples))
            new_samples_y = self.dist_y.rvs(len(invalid_samples))
            new_points = np.stack([new_samples_x, new_samples_y], axis=1)
            points[invalid_samples] = new_points

        return points

    def sample_with_redundency(self, n_samples=1, ratio=2, max_retries=5):
        """
        should be more efficient sampling. start by sampling more points (by ratio), and filter
        out points that are in masked areas this way the loop of resampling should be ran less time
        Resampling will happen up to max_retries times
        """

        if self.normalization_constant is None:
            self.normalization_constant = self._calculate_normalization_constant()

        samples_x = self.dist_x.rvs(ratio*n_samples)
        samples_y = self.dist_y.rvs(ratio*n_samples)
        points = np.stack([samples_x, samples_y], axis=1)

        valid_points = points[np.where(self.pdf(points) != 0)[0]]

        # make sure none of the points are in masked areas:
        retries = 0
        while len(valid_points) and retries < max_retries:
            retries += 1
            # print("resampling")
            new_samples_x = self.dist_x.rvs(ratio * n_samples)
            new_samples_y = self.dist_y.rvs(ratio * n_samples)
            new_points = np.stack([new_samples_x, new_samples_y], axis=1)
            new_valid_points = new_points[np.where(self.pdf(new_points) != 0)[0]]
            valid_points = np.concatenate((valid_points, new_valid_points))

            ratio *= 2

        return valid_points[:n_samples]

    def sample_max_points(self, n_samples, min_samples):
        """
        sample n_samples from the underlying gaussian distribution, remove samples in masked areas
        and don't sample again until you reach n_samples, just return what is filtered, unless
        there are less than min_samples in the filtered list, in which case do sample more
        """

        if self.normalization_constant is None:
            self.normalization_constant = self._calculate_normalization_constant()

        samples_x = self.dist_x.rvs(n_samples)
        samples_y = self.dist_y.rvs(n_samples)
        points = np.stack([samples_x, samples_y], axis=1)

        valid_points = points[np.where(self.pdf(points) != 0)[0]]

        if len(valid_points) < min_samples:
            valid_points = self.sample(n_samples=min_samples)

        return valid_points

