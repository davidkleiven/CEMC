import sys
import numpy as np
from itertools import combinations_with_replacement, filterfalse
from itertools import permutations
from scipy.optimize import minimize, curve_fit
from copy import deepcopy

class MultivariatePolynomial(object):
    def __init__(self, coeff=[], powers=[], cyclic=False):
        self.powers = powers
        self.coeff = coeff
        self.cyclic = cyclic

    def __call__(self, x):
        if len(x) != len(self.powers[0]):
            raise TypeError("The length of x needs to match!")

        def get_terms(powers):
            if self.cyclic:
                terms = [np.prod(np.power(x, np.roll(powers, i))) for i in range(3)]
            else:
                terms = [np.prod(np.power(x, powers))]
            return sum(terms)

        terms = list(map(get_terms, self.powers))
        return np.array(terms).dot(self.coeff)

    def deriv(self, x, dim=0):
        if len(x) != len(self.powers[0]):
            raise TypeError("The length of x needs to match!")

        def get_terms(powers):
            if self.cyclic:
                terms = []
                for i in range(3):
                    rolled_power = np.roll(powers, i)
                    terms.append(rolled_power[dim]*np.prod(np.power(x, rolled_power)))
            else:
                prefactors = deepcopy(powers)
                powers = list(powers)
                powers[dim] -= 1
                terms = [prefactors[dim]*np.prod(np.power(x, powers))]
            return sum(terms)

        terms = list(map(get_terms, self.powers))
        return np.array(terms).dot(self.coeff)
        



class TwoPhaseLandauPolynomial(object):
    def __init__(self, conc_order=2, strain_order=6, c1=0.0, c2=1.0,
                 num_dir=3):
        self.conc_order = conc_order
        self.strain_order = strain_order
        self.c1 = c1
        self.c2 = c2
        self.num_dir = num_dir
        self.coefficients = {
            "conc1": np.zeros(int(self.conc_order/2)),
            "conc2": np.zeros(int(self.conc_order/2)),
            "shape": np.zeros(self.num_strain_coeff)
        }
        self.optimal_shape_var = None

    @property
    def num_strain_coeff(self):
        return len(list(self.shape_powers)) - 1

    @property
    def shape_powers(self):
        exponents = range(0, self.strain_order+1, 2)

        def condition(x):
            return sum(x) > self.strain_order or sum(x) == 0
        return filterfalse(condition, combinations_with_replacement(exponents, r=self.num_dir))

    def conc1_terms(self, conc):
        powers = range(2, self.conc_order+1, 2)
        return map(lambda x: (conc-self.c1)**x, powers)

    def conc2_terms(self, conc):
        powers = range(2, self.conc_order+1, 2)
        return map(lambda x: (conc-self.c2)**x, powers)

    def shape_terms(self, shape):
        powers = list(self.shape_powers)
        def sum_shape_term(powers):
            terms = [np.prod(np.power(shape, np.roll(powers, i))) for i in range(3)]
            return sum(terms)
        return map(sum_shape_term, powers)

    def contribution_from_conc2(self, conc, shape):
        conc2_contrib = (conc - self.c2)*self.coefficients["conc2"][0]
        conc2_contrib *= np.sum(shape**2)
        return conc2_contrib

    def evaluate(self, conc, shape):
        """Evaluate the polynomial."""
        conc1_terms = list(self.conc1_terms(conc))
        conc1_contrib = np.array(conc1_terms).dot(self.coefficients["conc1"])

        conc2_contrib = self.contribution_from_conc2(conc, shape)
        shape_terms = list(self.shape_terms(shape))[1:]
        shape_contrib = np.array(shape_terms).dot(self.coefficients["shape"])
        return conc1_contrib + conc2_contrib + shape_contrib

    def fit(self, conc1, free_energy1, conc2, free_energy2, tol=1E-4, penalty=1E-8,
            fname=""):
        """Fit a free energy polynomial."""
        # First we fit to the first phase, where there are no
        # impact of the shape parameters
        terms = np.array(list(self.conc1_terms(conc1))).T
        coeff, _, _, _ = np.linalg.lstsq(terms, free_energy1, rcond=None)
        self.coefficients["conc1"] = coeff

        # Fit the second term
        conc1_terms = self.conc1_terms(conc2)
        conc1_contrib = np.zeros_like(conc2)
        for i, term in enumerate(conc1_terms):
            conc1_contrib += term*self.coefficients["conc1"][i]
        rhs = free_energy2 - conc1_contrib


        converged = False
        old_coeff = None
        init_shape = np.linspace(0.0, 1.0, len(conc2))
        self.optimal_shape_var = [init_shape[i]*np.ones(3) for i in range(len(conc2))]
        best_coeff = None
        best_rmse = 10000000.0
        best_optimal_slaved = None
        
        while not converged:
            terms = list(self.shape_terms(self.optimal_shape_var))
            terms = terms[1:]

            X = np.zeros((len(conc2), 2))
            X[:, 0] = 1
            X[:, 1] = [(conc2[i] - self.c2)*np.sum(shape**2)
                       for i, shape in enumerate(self.optimal_shape_var)]
            coeff, _, _, _ = np.linalg.lstsq(X, rhs)
            print(coeff)
            exit()


            for j in range(0, len(conc2)):
                terms =  list(self.shape_terms(self.optimal_shape_var[j]))
                X[j, 1:] = terms[1:]
            prec = (X.T.dot(X) + penalty*np.eye(X.shape[1]))
            coeff = np.linalg.inv(prec).dot(X.T.dot(rhs))
            self.coefficients["conc2"][0] = coeff[0]
            self.coefficients["shape"][:] = coeff[1:]

            self.optimize_shape_variables(conc2)

            if best_coeff is None:
                best_coeff = deepcopy(self.coefficients)
                best_optimal_slaved = deepcopy(self.optimal_shape_var)

            if old_coeff is None:
                old_coeff = coeff.copy()
            else:
                diff = np.max(np.abs(coeff - old_coeff))
                if  diff < tol:
                    converged = True
                old_coeff = coeff.copy()
                rmse = np.sqrt(np.mean((rhs - X.dot(coeff))**2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_coeff = deepcopy(self.coefficients)
                    best_optimal_slaved = deepcopy(self.optimal_shape_var)

                print("Difference: {}. Residual: {}".format(diff, rmse))
            self.coefficients = best_coeff

        if fname != "":
            slaved_params = np.array(best_optimal_slaved).T
            data = np.vstack((conc2, free_energy2, slaved_params)).T
            description = ", ".join(["param{}".format(x) for x in range(self.num_dir)])
            np.savetxt(fname, data, delimiter=",", header="Concentration, free energy" + description)

    def optimize_shape_variables(self, conc):
        """Optimize the shape variable given two concentrations."""

        for i, c in enumerate(conc):
            def optimize_cost(x):
                x_full = np.zeros(self.num_dir)
                x_full[:] = x
                return self.evaluate(c, x_full)

            res = minimize(optimize_cost, x0=self.optimal_shape_var[i][0],
                           bounds=[(0, 1)])
            self.optimal_shape_var[i] = np.ones(self.num_dir)*res["x"]
        
        


    
