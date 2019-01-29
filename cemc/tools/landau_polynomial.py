import sys
import numpy as np
from itertools import combinations_with_replacement, filterfalse
from itertools import permutations
from scipy.optimize import minimize, root, brentq
from copy import deepcopy

class TwoPhaseLandauPolynomial(object):
    def __init__(self, c1=0.0, c2=1.0, num_dir=3, init_guess=None):
        self.coeff = np.zeros(4)
        self.c1 = c1
        self.c2 = c2
        self.init_guess = init_guess

    def equil_shape_order(self, conc):
        if abs(self.coeff[3] < 1E-8):
            n_eq = -0.5*self.coeff[1]*(conc - self.c2)/self.coeff[2]
            print(conc, n_eq)
            if n_eq < 0.0:
                return 0.0
            return n_eq
        delta = (self.coeff[2]/(3.0*self.coeff[3]))**2 - \
            self.coeff[1]*(conc-self.c2)/(3.0*self.coeff[3])

        if delta < 0.0:
            return 0.0
        
        n_eq = -self.coeff[2]/(3.0*self.coeff[3]) + np.sqrt(delta)
        if n_eq < 0.0:
            return 0.0
        return n_eq

    def eval_at_equil(self, conc, on_off_center=False):
        """Evaluate the free energy at equillibrium order."""
        n_eq_sq = self.equil_shape_order(conc)
        return self.coeff[0]*(conc-self.c1)**2 + \
            self.coeff[1]*(conc - self.c2)*n_eq_sq + \
            self.coeff[2]*n_eq_sq**2 + \
            self.coeff[3]*n_eq_sq**3

    def fit(self, conc1, F1, conc2, F2):
        """Fit the free energy functional."""

        conc = np.concatenate((conc1, conc2))
        free_energy = np.concatenate((F1, F2))
        S1 = np.sum(F1*(conc1 - self.c1)**2)
        S2 = np.sum((conc1 - self.c1)**4)
        A = S1/S2
        remains = F2 - A*(conc2 - self.c1)**2
        
        S1 = np.sum(remains*(conc2 - self.c2))
        S2 = np.sum((conc2 - self.c2)**2)
        B = S1/S2

        S1 = np.sum((conc2 - self.c2))
        S2 = np.sum((conc2 - self.c2)**2)
        K = S1/S2
        C = -B/(2.0*K)

        if self.init_guess is not None:
            x0 = self.init_guess
        else:
            x0 = np.array([A, B, C, 15.0])

        def mse(x):
            self.coeff = x
            pred = [self.eval_at_equil(conc[i]) for i in range(len(conc))]
            pred = np.array(pred)
            mse = np.mean((pred - free_energy)**2)
            return mse

        res = minimize(mse, x0=x0)
        self.coeff = res["x"]


    
