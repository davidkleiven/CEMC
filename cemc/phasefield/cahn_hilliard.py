import numpy as np


class CahnHilliard(object):
    def __init__(self, degree=4, coeff=None):
        self.degree = degree
        self.coeff = coeff

    def fit(self, x, G):
        self.coeff = np.polyfit(x, G, self.degree)

    def evaluate(self, x):
        if self.coeff is None:
            raise ValueError("Coefficients is not set!")
        return np.polyval(self.coeff, x)
