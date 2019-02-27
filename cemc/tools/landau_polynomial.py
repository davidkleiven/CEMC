import numpy as np
from scipy.optimize import minimize, LinearConstraint
import scipy

SCIPY_VERSION = scipy.__version__


class TwoPhaseLandauPolynomial(object):
    """Class for fitting a Landau polynomial to free energy data

    :param float c1: Center concentration for the first phase
    :param float c2: Center concentration for the second phase
    :param np.ndarray init_guess: Initial guess for the parameters
        The polynomial fitting is of the form
        A*(x - c1)^2 + B*(x-c2)*y^2 + C*y^4 + D*y^6
        This array should therefore contain initial guess
        for the four parameter A, B, C and D.
    :param int conc_order1: Order of the polynomial in the first phase
    :param int conc_order2: Order of the polynomial in the second phase
    """
    def __init__(self, c1=0.0, c2=1.0, num_dir=3, init_guess=None,
                 conc_order1=2, conc_order2=2):
        self.coeff = np.zeros(conc_order2+3)
        self.conc_coeff = np.zeros(conc_order1+1)
        self.conc_order1 = conc_order1
        self.conc_order2 = conc_order2
        self.c1 = c1
        self.c2 = c2
        self.init_guess = init_guess
        self.boundary_coeff = None
        self.bounds = None

    def equil_shape_order(self, conc):
        """Calculate the equillibrium shape concentration.

        :param float conc: Concentration
        """

        if abs(self.coeff[-1] < 1E-8):
            n_eq = -0.5*self._eval_phase2(conc)/self.coeff[-2]
            if n_eq < 0.0:
                return 0.0
            return n_eq

        delta = (self.coeff[-2]/(3.0*self.coeff[-1]))**2 - \
            self._eval_phase2(conc)/(3.0*self.coeff[-1])

        if delta < 0.0:
            return 0.0

        n_eq = -self.coeff[-2]/(3.0*self.coeff[-1]) + np.sqrt(delta)
        if n_eq < 0.0:
            return 0.0
        return n_eq

    def equil_shape_order_derivative(self, conc):
        """Calculate the partial derivative of the equillibrium
            shape parameter with respect to the concentration.

        NOTE: This return the derivative of the square of of the
            order parameter with respect to the concentration.
        """

        delta = (self.coeff[-2]/(3.0*self.coeff[-1]))**2 - \
            self._eval_phase2(conc)/(3.0*self.coeff[-1])

        if delta < 0.0:
            return 0.0
        n_eq = self.equil_shape_order(conc)
        if n_eq <= 0.0:
            return 0.0
        p_der = np.polyder(self.coeff[:-2])
        return -0.5*np.polyval(p_der, conc-self.c2)/(3*np.sqrt(delta)*self.coeff[-1])

    def _eval_phase2(self, conc):
        """Evaluate the polynomial in phase2."""
        return np.polyval(self.coeff[:-2], conc - self.c2)

    def eval_at_equil(self, conc):
        """Evaluate the free energy at equillibrium order.

        :param float conc: Concentration
        """
        if self.bounds is not None:
            if conc < self.bounds[0]:
                return np.polyval(self.boundary_coeff[0], conc)
            elif conc > self.bounds[1]:
                return np.polyval(self.boundary_coeff[1], conc)

        n_eq_sq = self.equil_shape_order(conc)
        return np.polyval(self.conc_coeff, conc - self.c1) + \
            self._eval_phase2(conc)*n_eq_sq + \
            self.coeff[-2]*n_eq_sq**2 + \
            self.coeff[-1]*n_eq_sq**3

    def construct_end_of_domain_barrier(self):
        """Construct a barrier on the end of the domain."""
        coeff = []
        assert self.bounds is not None
        for i, lim in enumerate(self.bounds):
            # Lower limit
            value = self.evaluate(lim)
            deriv = self.partial_derivative(lim)
            matrix = np.zeros((3, 3))
            rhs = np.zeros(3)

            # Continuity
            matrix[0, 0] = lim**2
            matrix[0, 1] = lim
            matrix[0, 2] = 1.0
            rhs[0] = value

            # Smooth
            matrix[1, 0] = 2*lim
            matrix[1, 1] = 1.0
            rhs[1] = deriv

            # Max distance to minimum
            d = 0.01*(self.bounds[1] - self.bounds[0])
            if i == 0:
                # Lower limit
                matrix[2, 0] = 2*(lim - d)
            else:
                matrix[2, 0] = 2*(lim + d)
            matrix[2, 1] = 1.0
            new_coeff = np.linalg.solve(matrix, rhs)
            coeff.append(new_coeff)
        return coeff

    def evaluate(self, conc, shape=None):
        if self.bounds is not None:
            if conc < self.bounds[0]:
                return np.polyval(self.boundary_coeff[0], conc)
            elif conc > self.bounds[1]:
                return np.polyval(self.boundary_coeff[1], conc)

        if shape is None:
            return self.eval_at_equil(conc)

        return np.polyval(self.conc_coeff, conc - self.c1) + \
            self._eval_phase2(conc)*np.sum(shape**2) + \
            self.coeff[-2]*np.sum(shape**4) + \
            self.coeff[-1]*np.sum(shape**2)**3

    def partial_derivative(self, conc, shape=None, var="conc", direction=0):
        """Return the partial derivative with respect to variable."""
        allowed_var = ["conc", "shape"]
        if var not in allowed_var:
            raise ValueError("Variable has to be one of {}".format(allowed_var))

        if shape is None:
            shape = np.array([np.sqrt(self.equil_shape_order(conc))])

        if isinstance(shape, list):
            shape = np.array(shape)

        try:
            _ = shape[0]
        except (TypeError, IndexError):
            # Shape was a scalar, convert to array
            shape = np.array([shape])

        if var == "conc":
            if self.bounds and conc < self.bounds[0]:
                return np.polyval(np.polyder(self.boundary_coeff[0]), conc)
            elif self.bounds and conc > self.bounds[1]:
                return np.polyval(np.polyder(self.boundary_coeff[1]), conc)

            p1_der = np.polyder(self.conc_coeff)
            p2_der = np.polyder(self.coeff[:-2])
            return np.polyval(p1_der, conc-self.c1) + \
                np.polyval(p2_der, conc-self.c2)*np.sum(shape**2)

        elif var == "shape":
            return 2*self._eval_phase2(conc)*shape[direction] + \
                4*self.coeff[-2]*shape[direction]**3 + \
                3*self.coeff[-1]*np.sum(shape**2)**2 * 2*shape[direction]
        else:
            raise ValueError("Unknown derivative type!")

    def fit(self, conc1, F1, conc2, F2):
        """Fit the free energy functional.

        :param numpy.ndarray conc1: Concentrations in the first phase
        :param numpy.ndarray F1: Free energy in the first phase
        :param numpy.ndarray conc2. Concentrations in the second phase
        :param numpy.ndarray F2: Free energy in the second phase
        """
        conc = np.concatenate((conc1, conc2))
        free_energy = np.concatenate((F1, F2))
        self.conc_coeff = np.polyfit(conc1 - self.c1, F1, self.conc_order1)

        remains = F2 - np.polyval(self.conc_coeff, conc2 - self.c1)

        S1 = np.sum(remains*(conc2 - self.c2))
        S2 = np.sum((conc2 - self.c2)**2)
        B = S1/S2

        S1 = np.sum((conc2 - self.c2))
        S2 = np.sum((conc2 - self.c2)**2)
        K = S1/S2
        C = -B/(2.0*K)

        # Guess initial parameters
        mask = conc2 >= self.c2
        S1 = np.sum(remains[mask]*(conc2[mask] - self.c2))
        S2 = np.sum((conc2[mask] - self.c2)**2)
        B = S1/S2

        S1 = np.sum(remains*(conc2 - self.c2)**2)
        S2 = np.sum((conc2 - self.c2)**4)
        K = S1/S2
        C = - 0.5*B**2/K

        if self.init_guess is not None:
            x0 = self.init_guess
        else:
            x0 = np.array([B, C, min([abs(B), abs(C)])])

        def mse(x):
            self.coeff = x
            pred = [self.eval_at_equil(conc[i]) for i in range(len(conc))]
            pred = np.array(pred)
            mse = np.mean((pred - free_energy)**2)
            return mse

        if SCIPY_VERSION < '1.2.1':
            raise RuntimeError("Scipy version must be larger than 1.2.1!")

        num_constraints = 3
        A = np.zeros((num_constraints, len(self.coeff)))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)

        # Make sure last coefficient is positive
        A[0, -1] = 1.0
        lb[0] = 0.0
        ub[0] = np.inf
        cnst = LinearConstraint(A, lb, ub)

        # Make sure constant in polynomial 2 is zero
        A[1, -3] = 1.0
        lb[1] = -1E-16
        ub[1] = 1E-16

        # Make sure that the last coefficient is larger than
        # the secnod largest
        A[2, -1] = 1.0
        A[2, -2] = -1.0
        lb[2] = 0.0
        ub[2] = np.inf

        x0 = np.zeros(len(self.coeff))
        x0[-1] = 1.0
        x0[-4] = B
        x0[-2] = C
        x0[-1] = 1.0
        res = minimize(mse, x0=x0, method="SLSQP",
                       constraints=[cnst], options={"eps": 0.01})
        self.coeff = res["x"]
        self.bounds = [np.min(conc), np.max(conc)]
        self.boundary_coeff = self.construct_end_of_domain_barrier()

    def locate_common_tangent(self, loc1, loc2):
        """Locate a common tangent point in the vicinity of loc1 and loc2."""
        from scipy.optimize import newton

        def func(x):
            deriv1 = self.partial_derivative(x[0])
            deriv2 = self.partial_derivative(x[1])
            value1 = self.evaluate(x[0])
            value2 = self.evaluate(x[1])

            eq1 = deriv1 - deriv2
            eq2 = deriv1*(x[1] - x[0]) + value1 - value2
            return np.array([eq1, eq2])

        x0 = np.array([loc1, loc2])
        res = newton(func, x0, maxiter=5000)

        slope = self.partial_derivative(res[0])
        return res, slope
