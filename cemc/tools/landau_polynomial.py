import sys
import numpy as np
from itertools import combinations_with_replacement, filterfalse
from itertools import permutations
from scipy.optimize import minimize, root, brentq
from copy import deepcopy

class MultivariatePolynomial(object):
    def __init__(self, coeff=[], powers=[], cyclic=False, center=0.0):
        self.powers = powers
        self._coeff = coeff
        self.cyclic = cyclic
        self.center = 0.0

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        if len(new_coeff) != len(self._coeff):
            raise ValueError("New coefficients has the wrong length!")
        self._coeff = new_coeff

    def get_all_terms(self, x):
        try:
            n = len(x)
        except:
            x = np.array([x])
            n = len(x)

        if n != len(self.powers[0]):
            raise TypeError("The length of x needs to match!")

        def get_terms(powers):
            if self.cyclic:
                terms = [np.prod(np.power(x-self.center, np.roll(powers, i))) for i in range(3)]
            else:
                terms = [np.prod(np.power(x-self.center, powers))]
            return sum(terms)

        return list(map(get_terms, self.powers))

    def __call__(self, x):
        x = np.array(x)
        terms = self.get_all_terms(x)
        return np.array(terms).dot(self._coeff)

    def deriv(self, x, dim=0):
        try:
            n = len(x)
        except:
            x = np.array([x])
            n = len(x)

        if n != len(self.powers[0]):
            raise TypeError("The length of x needs to match!")

        x = np.array(x)
        def get_terms(powers):
            if self.cyclic:
                terms = []
                for i in range(3):
                    rolled_power = np.roll(powers, i)
                    terms.append(rolled_power[dim]*np.prod(np.power(x-self.center, rolled_power)))
            else:
                prefactors = deepcopy(powers)
                powers = list(powers)
                powers[dim] -= 1
                if powers[dim] < 0:
                    assert prefactors[dim] == 0
                    powers[dim] = 0
                terms = [prefactors[dim]*np.prod(np.power(x-self.center, powers))]
            return sum(terms)

        terms = list(map(get_terms, self.powers))
        return np.array(terms).dot(self._coeff)

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

def fit_polynomial(multipoly, x, y):
    """Fit the coefficient of a Multivariate Polynomial.
    
    :param MultivariatePolynomial multipoly: Instance of a Multivariate Polynomial
    :param numpy.ndarray x: X-parameters (shape NxM) where M is the
        number of free parameters and N is the number of datapoints
    :param numpy.ndarray y: Datapoints length N
    """
    num_terms = len(multipoly.coeff)
    num_pts = len(y)

    X = np.zeros((num_pts, num_terms))
    for i in range(len(y)):
        X[i, :] = multipoly.get_all_terms(x[i, :])

    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    multipoly.coeff = coeff
        
        


    
