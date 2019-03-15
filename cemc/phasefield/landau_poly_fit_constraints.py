import numpy as np
from scipy.signal import argrelextrema
from itertools import product, filterfalse


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

    def status_msg(self):
        return ""


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
        self.max_grad = 0.0

    def __call__(self, landau):
        gradient = np.zeros((self.eta.shape[0], 3))
        for i in range(3):
            gradient[:, i] = landau.partial_derivative(
                self.conc, shape=self.eta, var="shape", direction=i)

        # Find the component along the normal
        normal_comp = gradient.dot(self.normal)
        self.max_grad = np.max(np.abs(normal_comp))
        return self.weight*np.sum(normal_comp**2)

    def status_msg(self):
        return "Max. normal: {:.2e}".format(self.max_grad)


class InteriorMinima(LandauPolynomialFitConstraint):
    def __init__(self, weight=0.0, conc=1.0, eta_min=0.0, eta_max=1.0, num_eta=20):
        self.weight = weight
        self.conc = conc
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.num_eta = num_eta
        self.num_minima = 0

    def find_extrema(self, data):
        extremas = []

        for axis in range(len(data.shape)):
            extrema = argrelextrema(data, np.less_equal, axis=axis, order=3)
            extremas.append(set(zip(extrema[0].tolist(), extrema[1].tolist())))
        minima = set.intersection(*extremas)
        minima = filterfalse(lambda item: any(x == 0 or x == data.shape[0] - 1 or x == data.shape[1] - 1 for x in item), minima)
        return list(minima)

    def __call__(self, landau):
        F = np.zeros((self.num_eta, self.num_eta))
        n = np.linspace(self.eta_min, self.eta_max, self.num_eta)
        for indx in product(range(self.num_eta), repeat=2):
            F[indx] = landau.evaluate(self.conc, shape=[n[indx[0]], n[indx[1]], 0.0])
        minima = self.find_extrema(F)
        self.num_minima = len(minima)
        return self.weight*self.num_minima

    def status_msg(self):
        return "Num. interior minima: {}".format(self.num_minima)
