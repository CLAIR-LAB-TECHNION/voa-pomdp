import numpy as np
from scipy.stats import truncnorm
from modeling.belief.rectangles_overlap_resolution import resolve_overlaps, add_rectangle_to_decomposed
from line_profiler_pycharm import profile


def get_truncnorm_distribution(bounds, mu, sigma):
    a, b = (bounds[0] - mu) / sigma, (bounds[1] - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)


class UnnormalizedMasked2DTruncNorm:
    """
     Unormalized version, useful for action sampling.
     This is way faster since computing normalization factor and
     resolving masks overlaps takes most of the time.
    """

    def __init__(self, bounds_x, bounds_y, mu_x, sigma_x, mu_y, sigma_y,
                 build_xy_distributions=True):
        """
        if build_xy_distributions is False, the distributions are not built,
        and this object is not usable until .dist_x and .dist_y are initialized.
        This is useful for copying...
        """
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y

        self.masked_areas = []  # [[x,x] [y,y]] of areas that the distribution is zero
        self.bounds_list = []  # List to maintain added bounds
        if build_xy_distributions:
            self.dist_x = get_truncnorm_distribution(bounds_x, mu_x, sigma_x)
            self.dist_y = get_truncnorm_distribution(bounds_y, mu_y, sigma_y)
        else:
            self.dist_x = None
            self.dist_y = None

    def add_masked_area(self, mask_bounds):
        self.masked_areas.append(mask_bounds)

    def _check_if_bounds_list_legal(self, bounds_list):
        """
        check if there's intersection between bounds. If not, that's illegal because the whole space
        has probability of zero
        """
        if not bounds_list:
            return False
        if len(bounds_list) == 1:
            return True

        # Find the overall intersection bounds
        x_min = max(bound[0][0] for bound in bounds_list)
        y_min = max(bound[1][0] for bound in bounds_list)
        x_max = min(bound[0][1] for bound in bounds_list)
        y_max = min(bound[1][1] for bound in bounds_list)

        return x_min < x_max and y_min < y_max

    def add_new_bounds(self, bounds):
        if not self._check_if_bounds_list_legal(self.bounds_list + [bounds]):
            return

        new_bounds_x, new_bounds_y = bounds
        self.bounds_list.append(bounds)

        new_masked_areas = [
            [[self.bounds_x[0], new_bounds_x[0]], self.bounds_y],
            [[new_bounds_x[1], self.bounds_x[1]], self.bounds_y],
            [new_bounds_x, [self.bounds_y[0], new_bounds_y[0]]],
            [new_bounds_x, [new_bounds_y[1], self.bounds_y[1]]],
        ]
        self.masked_areas.extend(new_masked_areas)

    def add_multiple_new_areas(self, masked_areas, new_bounds):
        for bound in new_bounds:
            self.add_new_bounds(bound)
        self.masked_areas.extend(masked_areas)

    def update_parameters(self, mu_x, sigma_x, mu_y, sigma_y):
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y

        self.dist_x = get_truncnorm_distribution(self.bounds_x, mu_x, sigma_x)
        self.dist_y = get_truncnorm_distribution(self.bounds_y, mu_y, sigma_y)

    def pdf(self, points):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        x = points[:, 0]
        y = points[:, 1]

        result = self.dist_x.pdf(x) * self.dist_y.pdf(y)
        for mask in self.masked_areas:
            mask_condition = (mask[0][0] <= x) & (x <= mask[0][1]) & (mask[1][0] <= y) & (y <= mask[1][1])
            result[mask_condition] = 0
        return result

    def gaussian_pdf(self, points):
        """
         compute pdf of gaussian with same mu and sigma. This is not the correct pdf due to not normalizing,
         but it's much faster and useful for sampling
        """
        pdf_x = np.exp(-((points[:, 0] - self.mu_x) ** 2) / (2 * self.sigma_x ** 2)) \
                / (self.sigma_x * np.sqrt(2 * np.pi))
        pdf_y = np.exp(-((points[:, 1] - self.mu_y) ** 2) / (2 * self.sigma_y ** 2)) \
                / (self.sigma_y * np.sqrt(2 * np.pi))

        return pdf_x * pdf_y

    def are_points_masked(self, points, validate_in_workspace=True):
        """
        for array of points return boolean array of whether they are masked
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if len(self.masked_areas) == 0:
            return np.zeros(len(points), dtype=bool)

        x = points[:, 0]
        y = points[:, 1]
        masked = np.zeros_like(x, dtype=bool)
        for mask in self.masked_areas:
            mask_condition = (mask[0][0] <= x) & (x <= mask[0][1]) & (mask[1][0] <= y) & (y <= mask[1][1])
            masked |= mask_condition

        if validate_in_workspace:
            masked |= (x < self.bounds_x[0]) | (x > self.bounds_x[1]) | (y < self.bounds_y[0]) | (y > self.bounds_y[1])

        return masked

    def sample(self, n_samples=1):
        samples_x = self.dist_x.rvs(n_samples)
        samples_y = self.dist_y.rvs(n_samples)
        points = np.stack([samples_x, samples_y], axis=1)

        while True:
            invalid_samples = np.where(self.pdf(points) == 0)[0]
            if len(invalid_samples) == 0:
                break

            new_samples_x = self.dist_x.rvs(len(invalid_samples))
            new_samples_y = self.dist_y.rvs(len(invalid_samples))
            new_points = np.stack([new_samples_x, new_samples_y], axis=1)
            points[invalid_samples] = new_points

        return points

    @profile
    def _sample_from_truncnorm(self, max_n_samples, bounds_x=None, bounds_y=None):
        if bounds_x is None:
            samples_x = self.dist_x.rvs(max_n_samples)
        else:
            ax, bx = (bounds_x[0] - self.mu_x) / self.sigma_x, (bounds_x[1] - self.mu_x) / self.sigma_x
            samples_x = truncnorm.rvs(ax, bx, loc=self.mu_x, scale=self.sigma_x, size=max_n_samples)

        if bounds_y is None:
            samples_y = self.dist_y.rvs(max_n_samples)
        else:
            ay, by = (bounds_y[0] - self.mu_y) / self.sigma_y, (bounds_y[1] - self.mu_y) / self.sigma_y
            samples_y = truncnorm.rvs(ay, by, loc=self.mu_y, scale=self.sigma_y, size=max_n_samples)

        points = np.stack([samples_x, samples_y], axis=1)
        return points

    def sample_from_gaussian(self, n_samples=1):
        samples_x = np.random.normal(self.mu_x, self.sigma_x, n_samples)
        samples_y = np.random.normal(self.mu_y, self.sigma_y, n_samples)
        points = np.stack([samples_x, samples_y], axis=1)
        return points

    @profile
    def very_fast_sample(self, n_samples=1, ratio=2, max_retries=5, return_pdfs=True):
        """
        should be more efficient sampling. start by sampling more points (by ratio), and filter
        out points that are in masked areas this way the loop of resampling should be ran less time.
        Resampling will happen up to max_retries times
        """
        bounds_x, bounds_y = None, None
        if len(self.bounds_list) > 0:
            # there are bounds we know the block is in. it's smarter to sample there and not
            # in the entire space.
            # find intersection of bounds:
            x_min = max(bound[0][0] for bound in self.bounds_list)
            y_min = max(bound[1][0] for bound in self.bounds_list)
            x_max = min(bound[0][1] for bound in self.bounds_list)
            y_max = min(bound[1][1] for bound in self.bounds_list)

            # Bounded areas are often so small we can sample from uniform in them
            points_x = np.random.uniform(x_min, x_max, int(ratio * n_samples))
            points_y = np.random.uniform(y_min, y_max, int(ratio * n_samples))
            points = np.stack([points_x, points_y], axis=1)
        elif self.mu_x > 1 or self.mu_y > 1:
            # no bounds, but most of the samples from normal gaussian will be outside workspace, thus
            # it's better to sample from truncated normal...
            points = self._sample_from_truncnorm(int(ratio * n_samples), bounds_x, bounds_y)
            # TODO maybe sample from uniform?
        else:
            points = self.sample_from_gaussian(int(ratio * n_samples))  # This is x30 faster than truncnorm

        are_masked = self.are_points_masked(points, validate_in_workspace=True)
        valid_points = points[are_masked == 0]
        pdfs = self.gaussian_pdf(valid_points) if return_pdfs else None

        retries = 0
        while len(valid_points) < n_samples and retries < max_retries:
            retries += 1
            if len(self.bounds_list) > 0 or self.mu_x > 1 or self.mu_y > 1:
                new_points = self._sample_from_truncnorm(int(ratio * n_samples), bounds_x, bounds_y)
            else:
                new_points = self.sample_from_gaussian(int(ratio * n_samples))
            new_are_masked = self.are_points_masked(new_points, validate_in_workspace=True)
            new_valid_points = new_points[new_are_masked == 0]
            new_valid_pdfs = self.gaussian_pdf(new_valid_points) if return_pdfs else None

            valid_points = np.concatenate((valid_points, new_valid_points))
            pdfs = np.concatenate((pdfs, new_valid_pdfs))

            ratio *= ratio

        if return_pdfs:
            return valid_points[:n_samples], pdfs[:n_samples]
        return valid_points[:n_samples]


class Masked2DTruncNorm(UnnormalizedMasked2DTruncNorm):
    def __init__(self, bounds_x, bounds_y, mu_x, sigma_x, mu_y, sigma_y):
        super().__init__(bounds_x, bounds_y, mu_x, sigma_x, mu_y, sigma_y)
        self.non_overlapping_masked_areas = []
        self.normalization_constant = 1

    def add_masked_area(self, mask_bounds):
        super().add_masked_area(mask_bounds)
        self.non_overlapping_masked_areas = resolve_overlaps(self.masked_areas)
        self.normalization_constant = self._calculate_normalization_constant()

    def add_new_bounds(self, bounds):
        super().add_new_bounds(bounds)
        self.non_overlapping_masked_areas = resolve_overlaps(self.masked_areas)
        self.normalization_constant = self._calculate_normalization_constant()

    def add_multiple_new_areas(self, masked_areas, new_bounds):
        super().add_multiple_new_areas(masked_areas, new_bounds)
        if len(self.masked_areas) > 0:
            self.non_overlapping_masked_areas = resolve_overlaps(self.masked_areas)
            self.normalization_constant = self._calculate_normalization_constant()

    def update_parameters(self, mu_x, sigma_x, mu_y, sigma_y):
        super().update_parameters(mu_x, sigma_x, mu_y, sigma_y)
        self.normalization_constant = self._calculate_normalization_constant()

    def _calculate_normalization_constant(self):
        full_probability = 1
        for mask in self.non_overlapping_masked_areas:
            masked_probability = self._calculate_masked_probability(mask, self.bounds_x, self.mu_x, self.sigma_x,
                                                                    self.bounds_y, self.mu_y, self.sigma_y)
            full_probability -= masked_probability

        return max(full_probability, 1e-9)

    def _calculate_probability_mass(self, mask_dim, bounds, mu, sigma):
        start, end = mask_dim
        return truncnorm.cdf(end, (bounds[0] - mu) / sigma, (bounds[1] - mu) / sigma, loc=mu, scale=sigma) \
            - truncnorm.cdf(start, (bounds[0] - mu) / sigma, (bounds[1] - mu) / sigma, loc=mu, scale=sigma)

    def _calculate_masked_probability(self, mask, bounds_x, mu_x, sigma_x, bounds_y, mu_y, sigma_y):
        prob_mass_x = self._calculate_probability_mass(mask[0], bounds_x, mu_x, sigma_x)
        prob_mass_y = self._calculate_probability_mass(mask[1], bounds_y, mu_y, sigma_y)
        return prob_mass_x * prob_mass_y

    def pdf(self, points):
        unnormalized_pdf = super().pdf(points)
        return unnormalized_pdf / self.normalization_constant

    @profile
    def create_unnormalized(self):
        unnormalized = UnnormalizedMasked2DTruncNorm(self.bounds_x, self.bounds_y, self.mu_x,
                                                     self.sigma_x, self.mu_y, self.sigma_y,
                                                     build_xy_distributions=False)

        # This is ok to reference  and not copy as it's not going to change in the unnormalized
        # since it's only used for rollouting
        unnormalized.dist_x = self.dist_x
        unnormalized.dist_y = self.dist_y

        unnormalized.masked_areas = self.non_overlapping_masked_areas.copy()
        unnormalized.bounds_list = self.bounds_list.copy()
        return unnormalized

    def very_fast_sample(self, n_samples=1, ratio=1.5, max_retries=5, return_pdfs=True):
        raise NotImplementedError("This is not implemented for normalized version because it's optimized")