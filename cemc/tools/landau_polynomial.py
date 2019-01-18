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
    def __init__(self, conc_order=2, strain_order=6, c1=0.0, c2=1.0,
                 num_dir=3):
        self.conc_order = conc_order
        self.strain_order = strain_order
        self.c1 = c1
        self.c2 = c2
        self.num_dir = num_dir
        n = int(conc_order/2)
        cnc_pow = [(x,) for x in range(2, conc_order+1, 2)]
        shape_pow = np.array(list(self.shape_powers))/2
        self.polynomials = {
            "phase1": MultivariatePolynomial(coeff=np.zeros(n), powers=cnc_pow, center=c1),
            "phase2": MultivariatePolynomial(coeff=np.zeros(1), powers=[(1,)], center=c2),
            "shape": MultivariatePolynomial(coeff=np.zeros(self.num_strain_coeff), 
                                            powers=shape_pow, cyclic=True)
        }
        self.optimal_shape_var = None

    @property
    def num_strain_coeff(self):
        return len(list(self.shape_powers))

    @property
    def shape_powers(self):
        exponents = range(0, self.strain_order+1, 2)

        def condition(x):
            return sum(x) > self.strain_order or sum(x) <= 2
        return filterfalse(condition, combinations_with_replacement(exponents, r=self.num_dir))

    def evaluate(self, conc, shape):
        """Evaluate the polynomial."""
        return self.polynomials["phase1"](conc) + \
            self.polynomials["phase2"](conc)*np.sum(np.array(shape)**2) + \
            self.polynomials["shape"](np.array(shape)**2)

    def fit(self, conc1, free_energy1, conc2, free_energy2, tol=1E-4, penalty=1E-8,
            fname=""):
        """Fit a free energy polynomial."""
        # First we fit to the first phase, where there are no
        # impact of the shape parameters
        if len(conc1.shape) == 1:
            conc1 = conc1.reshape((conc1.shape[0], 1))
            assert len(conc1.shape) == 2
        fit_polynomial(self.polynomials["phase1"], conc1, free_energy1)

        # Fit the remaining coefficients
        # Define some functions to be used during the optimization
        def derivative_wrt_shape(x, conc):
            full_shape = np.array([x, x, x])
            deriv = self.polynomials["shape"].deriv(full_shape**2)
            return deriv + self.polynomials["phase2"](conc)

        conc1_contrib = np.array([self.polynomials["phase1"](c) for c in list(conc2)])
        rhs = free_energy2 - conc1_contrib
        # Cost function to minimize
        def cost_func(x):
            self.polynomials["phase2"].coeff[0] = x[0]
            self.polynomials["shape"].coeff = x[1:]
            
            # Optimize the cost function
            pred = np.zeros(len(conc2))
            for i in range(len(conc2)):
                #res = root(derivative_wrt_shape, 0.2, args=(conc2[i],))
                try:
                    shape = brentq(derivative_wrt_shape, 0.0, 1.0, args=(conc2[i],))
                except Exception:
                    shape = 1.0
                full_shape = [shape, shape, shape]
                pred[i] = self.evaluate(conc2[i], full_shape)
            mse = np.sum((pred-rhs)**2)
            #print(np.sqrt(mse), x, pred[0], rhs[0], pred[-1], rhs[-1])
            return mse

        options = {"eps": 0.05}
        x0 = -np.ones(len(self.polynomials["phase2"].coeff) + len(self.polynomials["shape"].coeff))

        def callback_log(x):
            print(x)
        res = minimize(cost_func, x0, options=options, method="BFGS", callback=callback_log)
        self.polynomials["phase2"].coeff[0] = res["x"][0]
        self.polynomials["shape"].coeff = res["x"][1:]

       

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
        
        


    
