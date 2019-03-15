import numpy as np


class GradCoeffEvaluator(object):
    def __init__(self):
        pass

    def evaluate(self, x, free_var):
        raise NotImplementedError("Has to be implemented in derived classes!")

    def derivative(self, x, free_var):
        raise NotImplementedError("Has to be implemented in derived classes!")


class SlavedTwoPhaseLandauEvaluator(GradCoeffEvaluator):
    """
    Example of a special evaluator indended to be used for
    free energy functions that can be described by
    TwoPhaseLandauPolynomial.

    :param TwoPhaseLandauPolynomial poly: Polynomial describing the free energy
    """
    def __init__(self, poly):
        from cemc.tools import TwoPhaseLandauPolynomial
        GradCoeffEvaluator.__init__(self)

        if not isinstance(poly, TwoPhaseLandauPolynomial):
            raise TypeError("poly has to be an instance of "
                            "TwoPhaseLandauPolynomial")
        self.poly = poly

    def evaluate(self, x, free_var):
        if free_var == 0:
            # The composition vary
            N = len(x[0])
            return np.array(self.poly.evaluate(x[0]))
        else:
            # The composition does not vary.
            # We assume that we have one unknown
            # parameter that is given by the mirrored
            # function of the free variable

            # Make sure that there is only one unknown variable
            assert sum(1 for item in x if item is None) == 1

            shape = np.zeros((len(x[free_var]), 3))
            shape[:, 1] = x[free_var]
            shape[:, 0] = shape[::-1, 1]
            N = len(x[free_var])
            return np.array(self.poly.evaluate(x[0], shape=shape))

    def derivative(self, x, free_var):
        N = len(x[free_var])
        unknown = 0
        for i, item in enumerate(x):
            if item is None:
                unknown = i
        assert unknown != free_var

        result = np.zeros((len(x), N))
        if free_var == 0:
            result[0, :] = 1.0
            eta_deriv = np.array(self.poly.equil_shape_order_derivative(x[0]))
            # eta_deriv[eta_deriv > 0.0] /= (2.0*eta_deriv[eta_deriv > 0.0])
            result[unknown, :] = eta_deriv
            return result
        else:
            # The composition don't vary so we assume mirrored
            # functions
            result[free_var, :] = 1.0
            result[unknown, :] = -1.0
            return result

