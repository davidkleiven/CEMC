import numpy as np


class CahnHilliard(object):
    def __init__(self, degree=4, coeff=None, bounds=None,
                 penalty=1E5, range_frac=0.1):
        self.degree = degree

        if coeff is not None:
            if len(coeff) != self.degree + 1:
                raise ValueError("Inconsistent arguments degree {}, "
                                 "num. coeff {}. The number of coefficients "
                                 "has to be degree+1"
                                 "".format(degree, len(coeff)))
        self.coeff = coeff
        self.bounds = bounds
        self.penalty = penalty
        self.rng_frac = range_frac

    def _regularization(self, x):
        if self.bounds is None:
            return 0.0

        if x > self.bounds[1]:
            return np.exp((x - self.bounds[1])/(self.rng_frac*self.conc_range))
        elif x < self.bounds[0]:
            return np.exp((self.bounds[0] - x)/(self.rng_frac*self.conc_range))
        return 0.0

    @property
    def conc_range(self):
        return abs(self.bounds[1] - self.bounds[0])

    def _regularization_deriv(self, x):
        if self.bounds is None:
            return 0.0

        if x > self.bounds[1]:
            return self._regularization(x)/(self.rng_frac*self.conc_range)
        elif x < self.bounds[0]:
            return -self._regularization(x)/(self.rng_frac*self.conc_range)
        return 0.0

    def fit(self, x, G):
        self.coeff = np.polyfit(x, G, self.degree)

    def evaluate(self, x):
        if self.coeff is None:
            raise ValueError("Coefficients is not set!")
        return np.polyval(self.coeff, x) + self.penalty*self._regularization(x)

    def deriv(self, x):
        """Return the first derivative of the polynomial."""
        der = np.polyder(self.coeff)
        return np.polyval(der, x) + self.penalty*self._regularization_deriv(x)
