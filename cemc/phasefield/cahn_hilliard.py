import numpy as np


class CahnHilliard(object):
    def __init__(self, degree=4, coeff=None):
        self.degree = degree

        if coeff is not None:
            if len(coeff) != self.degree + 1:
                raise ValueError("Inconsistent arguments degree {}, "
                                 "num. coeff {}. The number of coefficients "
                                 "has to be degree+1"
                                 "".format(degree, len(coeff)))
        self.coeff = coeff

    def fit(self, x, G):
        self.coeff = np.polyfit(x, G)

    def evaluate(self, x):
        if self.coeff is None:
            raise ValueError("Coefficients is not set!")
        return np.polyval(self.coeff, x)

    def deriv(self, x):
        """Return the first derivative of the polynomial."""
        p_der = np.polyder(self.coeff)
        return np.polyval(p_der, x)
