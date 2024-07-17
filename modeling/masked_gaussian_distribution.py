import numpy as np
from scipy.stats import truncnorm
from modeling.rectangles_overlap_resolution import resolve_overlaps


class Masked2DTruncNorm:
    def __init__(self, bounds_x, bounds_y, mu_x, sigma_x, mu_y, sigma_y):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y

        self.masked_areas = []
        ax, bx = (bounds_x[0] - mu_x) / sigma_x, (bounds_x[1] - mu_x) / sigma_x
        self.dist_x = truncnorm(ax, bx, loc=mu_x, scale=sigma_x)
        ay, by = (bounds_y[0] - mu_y) / sigma_y, (bounds_y[1] - mu_y) / sigma_y
        self.dist_y = truncnorm(ay, by, loc=mu_y, scale=sigma_y)

        self.normalization_constant = None

    def add_masked_area(self, mask_bounds):
        # Add new mask and merge overlapping masks
        self.masked_areas.append(mask_bounds)
        self.masked_areas = resolve_overlaps(self.masked_areas)
        self.normalization_constant = self._calculate_normalization_constant()

    def _calculate_normalization_constant(self):
        """Calculates the normalization constant for the joint PDF."""
        full_probability = 1  # Assume full probability mass covers the entire normalized range
        for mask in self.masked_areas:
            masked_probability = self._calculate_masked_probability(mask, self.bounds_x, self.mu_x, self.sigma_x,
                                                                    self.bounds_y, self.mu_y, self.sigma_y)
            full_probability -= masked_probability
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

    def pdf(self, points):
        # Ensure normalization constant is calculated
        if self.normalization_constant is None:
            self.normalization_constant = self._calculate_normalization_constant()

        # Extract x and y coordinates
        points = np.asarray(points)
        x = points[:, 0]
        y = points[:, 1]

        # Vectorized check for masked areas
        result = (self.dist_x.pdf(x) * self.dist_y.pdf(y)) / self.normalization_constant
        for mask in self.masked_areas:
            mask_condition = (mask[0][0] <= x) & (x <= mask[0][1]) & (mask[1][0] <= y) & (y <= mask[1][1])
            result[mask_condition] = 0
        return result



