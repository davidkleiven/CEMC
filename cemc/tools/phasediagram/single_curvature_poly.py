from scipy.optimize import minimize
import numpy as np


class SingleCurvaturePolynomial(object):
    """Class for fitting polynomial with constraints

    """
    def __init__(self, curvature="convex", exp_scale=1.0, alpha=1000.0,
                 slope="none"):
        self.curvature = curvature
        self.exp_scale = exp_scale
        self.alpha = alpha
        self.slope = slope

    def fit(self, x, y, order=2):
        """Fit a polynomial with curvature in given direction."""
        def cost_func(coeff):
            pred = np.zeros_like(y)
            pred = np.polyval(coeff, x)
            rmse = np.sqrt(np.mean((pred - y)**2))

            p_der = np.polyder(coeff, m=2)
            curv = np.polyval(p_der, x)

            p_der = np.polyder(coeff, m=1)
            der = np.polyval(p_der, x)

            penalty = 0.0
            if self.curvature == "convex":
                value = np.min(curv)

                if value < 0.0:
                    penalty = np.exp(abs(value)*self.exp_scale) - 1.0
            else:
                value = np.max(curv)
                if value > 0.0:
                    penalty = np.exp(value*self.exp_scale) - 1.0

            if self.slope == "increasing":
                min_slope = np.min(der)
                if min_slope < 0.0:
                    penalty += np.exp(abs(min_slope)*self.exp_scale) - 1.0
            elif self.slope == "decreasing":
                max_slope = np.max(der)
                if max_slope > 0.0:
                    penalty += np.exp(max_slope*self.exp_scale) - 1.0

            return rmse + self.alpha*penalty

        coeff0 = np.polyfit(x, y, order)
        res = minimize(cost_func, x0=coeff0)
        coeff = res["x"]
        return coeff
