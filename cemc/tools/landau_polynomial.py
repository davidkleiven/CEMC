import numpy as np
from scipy.optimize import minimize, LinearConstraint, fsolve
from scipy.optimize import NonlinearConstraint
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
        self.conc_coeff2 = np.zeros(conc_order2+1)
        self.conc_coeff = np.zeros(conc_order1+1)
        self.coeff_shape = np.zeros(5)
        self.conc_order1 = conc_order1
        self.conc_order2 = conc_order2
        self.c1 = c1
        self.c2 = c2
        self.init_guess = init_guess
        self.num_dir = num_dir
        self.boundary_coeff = None
        self.bounds = None

    def equil_shape_order(self, conc):
        """Calculate the equillibrium shape concentration.

        :param float conc: Concentration
        """
        C = self.coeff_shape[0]
        D = self.coeff_shape[2]

        if abs(D < 1E-8):
            n_eq = -0.5*self._eval_phase2(conc)/C
            if n_eq < 0.0:
                return 0.0
            return n_eq

        delta = (C/(3.0*D))**2 - \
            self._eval_phase2(conc)/(3.0*D)

        if delta < 0.0:
            return 0.0

        n_eq = -C/(3.0*D) + np.sqrt(delta)
        if n_eq < 0.0:
            return 0.0
        return np.sqrt(n_eq)

    def equil_shape_order_derivative(self, conc):
        """Calculate the partial derivative of the equillibrium
            shape parameter with respect to the concentration.

        NOTE: This return the derivative of the square of of the
            order parameter with respect to the concentration.
        """

        C = self.coeff_shape[0]
        D = self.coeff_shape[2]

        delta = (C/(3.0*D))**2 - \
            self._eval_phase2(conc)/(3.0*D)

        if delta < 0.0:
            return 0.0
        n_eq = self.equil_shape_order(conc)
        if n_eq <= 0.0:
            return 0.0
        p_der = np.polyder(self.conc_coeff2)
        return -0.5*np.polyval(p_der, conc-self.c2) / \
            (3*np.sqrt(delta)*D*2*n_eq)

    def _eval_phase2(self, conc):
        """Evaluate the polynomial in phase2."""
        return np.polyval(self.conc_coeff2, conc - self.c2)

    def eval_at_equil(self, conc):
        """Evaluate the free energy at equillibrium order.

        :param float conc: Concentration
        """
        if self.bounds is not None:
            if conc < self.bounds[0]:
                return np.polyval(self.boundary_coeff[0], conc)
            elif conc > self.bounds[1]:
                return np.polyval(self.boundary_coeff[1], conc)

        n_eq = self.equil_shape_order(conc)
        return np.polyval(self.conc_coeff, conc - self.c1) + \
            self._eval_phase2(conc)*n_eq**2 + \
            self.coeff_shape[0]*n_eq**4 + \
            self.coeff_shape[2]*n_eq**6

    def construct_end_of_domain_barrier(self, f_range):
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

            # Positive curvature
            d = 0.1*(self.bounds[1] - self.bounds[0])
            matrix[2, 0] = 1.0
            rhs[2] = 0.01*f_range/d**2
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

        full_shape = np.zeros(3)
        full_shape[:len(shape)] = shape
        shape = full_shape

        return np.polyval(self.conc_coeff, conc - self.c1) + \
            self._eval_phase2(conc)*np.sum(shape**2) + \
            self.coeff_shape[0]*np.sum(shape**4) + \
            self.coeff_shape[1]*(shape[0]**2 * shape[1]**2 +
                                 shape[0]**2 * shape[2]**2 +
                                 shape[1]**2 * shape[2]**2) + \
            self.coeff_shape[2]*np.sum(shape**6) + \
            self.coeff_shape[3]*(shape[0]**4 * (shape[1]**2 + shape[2]**2) +
                                 shape[1]**4 * (shape[0]**2 + shape[2]**2) +
                                 shape[2]**4 * (shape[0]**2 + shape[1]**2)) + \
            self.coeff_shape[4]*np.prod(shape**2)

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

        full_shape = np.zeros(3)
        full_shape[:len(shape)] = shape
        shape = full_shape

        if var == "conc":
            if self.bounds and conc < self.bounds[0]:
                return np.polyval(np.polyder(self.boundary_coeff[0]), conc)
            elif self.bounds and conc > self.bounds[1]:
                return np.polyval(np.polyder(self.boundary_coeff[1]), conc)

            p1_der = np.polyder(self.conc_coeff)
            p2_der = np.polyder(self.conc_coeff2)
            return np.polyval(p1_der, conc-self.c1) + \
                np.polyval(p2_der, conc-self.c2)*np.sum(shape**2)

        elif var == "shape":
            d = direction
            return 2*self._eval_phase2(conc)*shape[d] + \
                4*self.coeff_shape[0]*shape[d]**3 + \
                2*self.coeff_shape[1]*shape[d]*(shape[(d+1) % 3] + shape[(d+2) % 3]) + \
                6*self.coeff_shape[2]*shape[d]**5 + \
                4*self.coeff_shape[3]*shape[d]**3*(shape[(d+1) % 3]**2 + shape[(d+2) % 3]**2) + \
                2*self.coeff_shape[3]*shape[d]*(shape[(d+1) % 3]**4 + shape[(d+2) % 3]**4) + \
                2*self.coeff_shape[4]*shape[d]*shape[(d+1) % 3]**2 * shape[(d+2) % 3]**2
        else:
            raise ValueError("Unknown derivative type!")

    def fit(self, conc1, F1, conc2, F2, minimum_at_ends=True):
        """Fit the free energy functional.

        :param numpy.ndarray conc1: Concentrations in the first phase
        :param numpy.ndarray F1: Free energy in the first phase
        :param numpy.ndarray conc2. Concentrations in the second phase
        :param numpy.ndarray F2: Free energy in the second phase
        """
        conc = np.concatenate((conc1, conc2))
        free_energy = np.concatenate((F1, F2))
        # self.conc_coeff = np.polyfit(conc1 - self.c1, F1, self.conc_order1)
        X = np.zeros((len(conc1), self.conc_order1+1))
        for power in range(self.conc_order1):
            X[:, power] = (conc1 - self.c1)**(self.conc_order1-power)
        y = F1.copy()
        indx = np.argmin((np.abs(conc1 - self.c2)))
        slope = (F1[indx] - F1[0])/(conc1[indx] - conc1[0])
        y[conc1 > self.c2] = y[indx]
        self.conc_coeff = np.linalg.lstsq(X, y)[0]

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
            self.conc_coeff2 = x[:self.conc_order2+1]
            self.coeff_shape[0] = x[-2]
            self.coeff_shape[2] = x[-1]
            pred = [self.evaluate(conc[i]) for i in range(len(conc))]
            pred = np.array(pred)

            mse = np.mean((pred - free_energy)**2)
            concs = np.linspace(0.0, self.c2, 10).tolist()
            n_eq = np.array([self.equil_shape_order(c) for c in concs])
            mse_eq = np.sum(n_eq**2)
            return mse + 10*mse_eq

        if SCIPY_VERSION < '1.2.1':
            raise RuntimeError("Scipy version must be larger than 1.2.1!")

        num_constraints = 3
        A = np.zeros((num_constraints, self.conc_order2+3))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)

        # Make sure last coefficient is positive
        A[0, -1] = 1.0
        lb[0] = 1.0
        ub[0] = np.inf
        cnst = LinearConstraint(A, lb, ub)

        # Make sure we have a local minimum at n_eq
        A[1, -1] = 3.0
        A[1, -2] = 1.0
        lb[1] = 0.0
        ub[1] = np.inf

        # Make sure that the last coefficient is larger than
        # the secnod largest
        A[2, -2] = 1.0
        lb[2] = -np.inf
        ub[2] = 0.0

        x0 = np.zeros(self.conc_order2+3)

        B = -10  # TODO: We need to be able to set this
        x0[-4] = B
        x0[-2] = C
        x0[-1] = 1.0

        # Add constraint on the equillibrium shape parameter
        def n_eq_cnst(x):
            old_conc_coeff2 = self.conc_coeff2.copy()
            old_coeff_shape = self.coeff_shape.copy()
            self.conc_coeff2 = x[:self.conc_order2]
            self.coeff_shape[0] = x[-2]
            self.coeff_shape[2] = x[-1]
            concs = np.linspace(0.0, self.c2, 10).tolist()
            n_eq = [self.equil_shape_order(c) for c in concs]

            # Reset the concentration
            self.conc_coeff2[:] = old_conc_coeff2
            self.coeff_shape[:] = old_coeff_shape
            return np.mean(n_eq)

        n_eq_cnst = NonlinearConstraint(n_eq_cnst, 0.0, 1E-8)

        res = minimize(mse, x0=x0, method="SLSQP",
                       constraints=[cnst], options={"eps": 0.01})
        self.conc_coeff2 = res["x"][:-2]
        self.coeff_shape[0] = res["x"][-2]
        self.coeff_shape[2] = res["x"][-1]
        self.bounds = [np.min(conc), np.max(conc)]
        f_range = np.max(free_energy) - np.min(free_energy)
        self.boundary_coeff = self.construct_end_of_domain_barrier(f_range)

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

    def slaved_gradient_coefficients(self, conc_grad_coeff, interface_energies,
                                     density):
        """Calculate the gradient coefficients based on slaved order
        parameters."""

        grad = []

        def equation(value, gamma):
            N = 100
            x = np.linspace(self.bounds[0], self.bounds[1], N)
            deriv = [self.equil_shape_order_derivative(x[i]) for i in range(N)]
            f = [self.evaluate(x[i]) for i in range(N)]
            f = np.array(f)
            deriv = np.array(deriv)
            f1 = f[0]
            f2 = f[-1]
            slope = (f2 - f1)/(x[-1] - x[0])
            interscept = f1 - slope*x[0]
            f -= (slope*x + interscept)
            f -= np.min(f)
            integrand = np.sqrt(f)*np.sqrt(conc_grad_coeff + value*deriv**2)
            dx = x[1] - x[0]
            return 2.0*density*np.trapz(integrand, dx=dx) - gamma

        for energy in interface_energies:
            res = fsolve(equation, conc_grad_coeff, args=(energy,))[0]
            grad.append(res)
        return grad

    def save_poly_terms(self, fname="pypolyterm.csv"):
        """Store the required arguments that can be used to 
            construct poly terms for phase field calculations."""
        num_terms = len(self.conc_coeff) + len(self.conc_coeff2) + len(self.coeff_shape)

        data = np.zeros((num_terms, self.num_dir+1+2))
        header = "Coefficient, Inner powers ()..., outer power"
        row = 0

        # Coefficients from the first phase
        for i in range(len(self.conc_coeff)):
            data[row, 0] = self.conc_coeff[i]
            data[row, 1] = len(self.conc_coeff) - i
            data[row, -1] = 1.0
            row += 1

        # Coefficient in the second phase
        coeff = self.conc_coeff2
        for i in range(len(coeff)):
            data[row, 0] = coeff[i]
            data[row, 1] = len(coeff) - i
            data[row, 2:-1] = 2.0
            data[row, -1] = 1.0
            row += 1

        # Pure shape parameter terms
        data[row, 0] = self.coeff_shape[0]
        data[row, 2:-1] = 4
        data[row, -1] = 1.0
        row += 1

        data[row, 0] = self.coeff_shape[2]
        data[row, 2:-1] = 2
        data[row, -1] = 3
        row += 1
        np.savetxt(fname, data, delimiter=",", header=header)
        print("Polynomial data written to {}".format(fname))

    def _equil_shape_fixed_conc_and_shape_intermediates(self, conc, shape, min_type):
        """Return helper quantities for the equillibrium shape."""
        K = self._eval_phase2(conc)
        K += self.coeff_shape[1]*shape**2
        K += self.coeff_shape[3]*shape**4

        Q = self.coeff_shape[0] + self.coeff_shape[3]*shape**2

        if min_type == "mixed":
            Q += 0.5*self.coeff_shape[1]
            Q += 0.5*self.coeff_shape[4]*shape**2

        D = 3.0*self.coeff_shape[2]
        return K, Q, D

    def equil_shape_fixed_conc_and_shape(self, conc, shape, min_type="pure"):
        """Return the equillibrium shape parameter.

        :param float conc: Concentration
        :param float shape: Shape parameter
        :param str min_type: Type of minimum. If pure, the third
            shape parameter is set to zero. If mixed, the two
            free shape parameters are required to be the same.
        """
        allowed_types = ["pure", "mixed"]

        if min_type not in allowed_types:
            raise ValueError("min_type has to be one of {}"
                             "".format(allowed_types))

        K, Q, D = self._equil_shape_fixed_conc_and_shape_intermediates(
            conc, shape, min_type)

        delta = (Q/D)**2 - K/D

        if delta < 0.0:
            return 0.0
        n_sq = -Q/D + np.sqrt(delta)

        if n_sq < 0.0:
            return 0.0
        return np.sqrt(n_sq)

    def equil_shape_fixed_conc_and_shape_deriv(self, conc, shape,
                                               min_type="pure"):
        """Differentiate with respect to the fixed shap parameter."""
        K, Q, D = self._equil_shape_fixed_conc_and_shape_intermediates(
            conc, shape, min_type)

        dQ_dn = 2*self.coeff_shape[3]*shape

        if min_type == "mixed":
            dQ_dn += 2*self.coeff_shape[4]*shape*0.5

        dK_dn = 2*self.coeff_shape[1]*shape + \
            4*self.coeff_shape[3]*shape**3

        n_eq = self.equil_shape_fixed_conc_and_shape(
            conc, shape, min_type=min_type)

        if n_eq <= 0.0:
            return 0.0

        delta = (Q/D)**2 - K/D
        deriv = - dQ_dn/D + 0.5*(2*Q*dQ_dn/D**2 - dK_dn/D)/np.sqrt(delta)
        return 0.5*deriv/n_eq

    def plot_individual_polys(self):
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        conc = np.linspace(0.0, 1.0, 100)
        ph1 = np.polyval(self.conc_coeff, conc)
        ax.plot(conc, ph1, label="Phase1")
        return fig

    def fit_fixed_conc_varying_eta(self, conc, eta, free_energy,
                                   path_type="pure"):
        """Perform fit at fixed composition, but varying eta.

        :param float conc: Fixed concentration
        :param array eta: Array with eta values
        :param array free_energy: Free energy densitties
        :parma str path_type: Determine how the remaining shape
            parameters varies. Has to be one of pure or mixed.
            If pure, it is assumed that the third shape parameter
            remains zero, while the second is given by its
            equilibrium value. If mixed, it assumed that
            both the second and the third are equal and
            equal to the equilibrium value.
        """

        def mse_function(x):
            self.coeff_shape[1] = x[0]
            self.coeff_shape[3:] = x[1:]
            eq_shape = [self.equil_shape_fixed_conc_and_shape(
                conc, n, min_type=path_type) for n in list(eta)]

            if path_type == "pure":
                pred = [self.evaluate(conc, shape=[n1, n2, 0.0])
                        for n1, n2 in zip(eta, eq_shape)]
            elif path_type == "mixed":
                pred = [self.evaluate(conc, shape=[n1, n2, n2])
                        for n1, n2 in zip(list(eta), eq_shape)]
            else:
                raise ValueError("Unknown path_type {}".format(path_type))

            pred = np.array(pred)
            mse = np.sum((pred - free_energy)**2)
            return mse

        # Last term has to be positive
        num_coeff = len(self.coeff_shape) - 2
        x0 = np.zeros(num_coeff)
        A = np.zeros((1, num_coeff))
        ub = np.zeros(1)
        lb = np.zeros(1)
        ub[0] = np.inf
        A[0, -1] = 1.0

        cnst = LinearConstraint(A, lb, ub)
        res = minimize(mse_function, x0, method="SLSQP", constraints=cnst)

        # Update the shape coefficients
        self.coeff_shape[1] = res["x"][0]
        self.coeff_shape[3:] = res["x"][1:]

    def _get_linear_bias(self, x, y):
        """Calculate the linear base line."""
        cut_indx = int(len(y)/2)
        indx_left = np.argmin(y[:cut_indx])
        indx_right = np.argmin(y[cut_indx:]) + cut_indx
        x_left = x[indx_left]
        x_right = x[indx_right]
        y_left = y[indx_left]
        y_right = y[indx_right]

        base_line = (y_right - y_left)*(x-x_left)/(x_right - x_left) + \
            y_left
        return base_line, indx_left, indx_right

    def fit_features_fixed_conc_varying_eta(self, conc, eta, free_energy,
                                   path_type="pure", opt_priority={}, verbose=True):
        """Perform fit at fixed composition, but varying eta.

        :param float conc: Fixed concentration
        :param array eta: Array with eta values
        :param array free_energy: Free energy densitties
        :parma str path_type: Determine how the remaining shape
            parameters varies. Has to be one of pure or mixed.
            If pure, it is assumed that the third shape parameter
            remains zero, while the second is given by its
            equilibrium value. If mixed, it assumed that
            both the second and the third are equal and
            equal to the equilibrium value.
        """

        # Locate the minimum near the left side of the curve
        base_line, indx_left, indx_right = self._get_linear_bias(eta, free_energy)

        form_free_energy = free_energy - base_line
        assert np.all(form_free_energy >= 0.0)

        integral = np.trapz(form_free_energy[indx_left:indx_right])
        integral_sqrt = np.trapz(np.sqrt(form_free_energy[indx_left:indx_right]))

        peak_pos = np.argmax(free_energy)
        peak_value = free_energy[peak_pos]

        # Extract optimization weights
        start_weight = opt_priority.get("start_weight", 200.0)
        smooth_weight = opt_priority.get("smooth", 1.0)
        integral_weight = opt_priority.get("integral", 0.0)
        peak_pos_weight = opt_priority.get("peak_pos", 0.0)
        peak_value_weight = opt_priority.get("peak_value", 0.0)

        def mse_function(x):
            self.coeff_shape[1] = x[0]
            self.coeff_shape[3:] = x[1:]
            eq_shape = [self.equil_shape_fixed_conc_and_shape(
                conc, n, min_type=path_type) for n in list(eta)]

            if path_type == "pure":
                pred = [self.evaluate(conc, shape=[n1, n2, 0.0])
                        for n1, n2 in zip(eta, eq_shape)]
            elif path_type == "mixed":
                pred = [self.evaluate(conc, shape=[n1, n2, n2])
                        for n1, n2 in zip(list(eta), eq_shape)]
            else:
                raise ValueError("Unknown path_type {}".format(path_type))

            # Calculate the integral
            pred = np.array(pred)
            base_line, indx_left, indx_right = self._get_linear_bias(eta, pred)
            form_pred = pred - base_line

            integral_pred = np.trapz(form_pred[indx_left:indx_right])

            peak_val_pred = np.max(pred)
            from scipy.ndimage.filters import laplace
            laplace_filter = laplace(pred)

            if verbose:
                print("Integral: {:.2E} ({:.2E}), Peak value {:.2E} ({:.2E} ),"
                      " Start value: {:.2E} ({:.2E})"
                      "".format(integral_pred, integral,
                                peak_val_pred, peak_value, pred[indx_right],
                                free_energy[indx_right]))
            return integral_weight*(integral_pred - integral)**2 + \
                peak_value_weight*(peak_val_pred - peak_value)**2 + \
                start_weight*(pred[indx_right] - free_energy[indx_right])**2 + \
                smooth_weight*(np.sum(laplace_filter**2))
                

        # Last term has to be positive
        num_coeff = len(self.coeff_shape) - 2
        x0 = np.zeros(num_coeff)
        A = np.zeros((1, num_coeff))
        ub = np.zeros(1)
        lb = np.zeros(1)
        ub[0] = np.inf
        A[0, -1] = 1.0

        cnst = LinearConstraint(A, lb, ub)
        res = minimize(mse_function, x0, method="SLSQP", constraints=cnst)

        # Update the shape coefficients
        self.coeff_shape[1] = res["x"][0]
        self.coeff_shape[3:] = res["x"][1:]
