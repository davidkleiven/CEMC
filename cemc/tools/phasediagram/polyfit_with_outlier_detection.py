import numpy as np
from scipy.stats import norm


class Polynomial(object):
    def __init__(self, order=2, conf=0.05):
        self.order = order
        self.confidence = conf

    def fit(self, x, y):
        """Fit polynomial to x, y data."""
        if len(x) != len(y):
            raise ValueError("x and y has to be the same length")

        if len(x) <= self.order+2:
            raise ValueError("The number of datapoints has to be at least polyorder+2")

        end = self.order+2
        finished = False
        while not finished:
            poly = np.polyfit(x[:end], y[:end], self.order)
            diff = np.polyval(poly, x[:end]) - y[:end]
            std = np.std(diff)

            n_indx = end + 1
            if n_indx >= len(x):
                return poly

            if self._belongs_to_sequence(poly, std, x[n_indx], y[n_indx]):
                end = next_indx
            else:
                finished = True
        return poly

    def _belongs_to_sequence(self, poly, std, x_new, y_new):
        """Return True if the next point can be predicted with."""
        predicted = np.polyval(poly, x_new)
        diff = predicted - y_new
        z = diff/(np.sqrt(2.0)*std)

        # Probability that z of the zero distribution is
        # larger than the value observed
        percentile = norm.ppf(self.confidence)
        return abs(z) < percentile
