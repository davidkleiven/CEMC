import numpy as np


class LandauPolynomialFitConstraint(object):
    """
    Generic base class for constraints applied to the Landau polynomials
    """
    def __init__(self):
        pass

    def __call__(self, landau):
        raise NotImplementedError("Has to be implemented in derived classes!")


class PeakPosition(LandauPolynomialFitConstraint):
    """
    Constraint that attempts to fix the position of the peak
    """
    def __init__(self, weight=0.0, peak_indx=0, conc=1.0, eta=[], free_eng=[]):
        self.weight = weight
        self.peak_indx = peak_indx
        self.eta = eta
        self.free_eng = free_eng
        self.conc = conc

    def __call__(self, landau):
        pred = np.array(landau.evaluate(self.conc, shape=self.eta))
        value_peak = pred[self.peak_indx]
        cost_peak = np.sum((pred[pred > value_peak] - value_peak)**2)
        return self.weight*cost_peak


class StraightLineSaddle(LandauPolynomialFitConstraint):
    """
    Constraint that attempts to put the gradient normal to a straight
    line to zero.
    """
    def __init__(self, weight=0.0, normal=[1.0, 0.0, 0.0], eta=[], conc=0.0):
        self.weight = weight
        self.normal = np.array(normal)
        self.normal /= np.sqrt(self.normal.dot(self.normal))
        self.eta = eta
        self.conc = conc

    def __call__(self, landau):
        gradient = np.zeros((self.eta.shape[0], 3))
        for i in range(3):
            gradient[:, i] = landau.partial_derivative(
                self.conc, shape=self.eta, var="shape", direction=i)

        # Find the component along the normal
        normal_comp = gradient.dot(self.normal)
        return self.weight*np.sum(normal_comp**2)