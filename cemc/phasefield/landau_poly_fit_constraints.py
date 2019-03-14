import numpy as np

class LandauPolynomialFitConstraint(object):
    def __init__(self):
        pass

    def __call__(self, landau):
        raise NotImplementedError("Has to be implemented in derived classes!")


class PeakPosition(LandauPolynomialFitConstraint):
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