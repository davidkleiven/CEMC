import numpy as np


class BinaryCriticalPoints(object):
    """
    Class for finding critical points on for binary phase
    diagrams.
    """
    def __init__(self):
        pass

    def coexistence_points(self, phase1, phase2):
        """
        Find coexistence points.

        :param np.ndarray phase1: Second order polynomial for phase1
        :param np.ndarray phase2: Second order polynomial for phase2
        """

        delta_a = phase2[2] - phase1[2]
        delta_b = phase2[1] - phase1[1]
        c1 = phase1[0]
        c2 = phase2[0]

        # Calculate points that enters in second order
        # equation (x - B)**2 = C
        B = 0.5*c2*delta_b/(c1*c2 - c2**2)
        C = (4*c1*delta_a + delta_b**2)/(4*c1*c2 - 4*c2**2)

        x2_minus = B - np.sqrt(B**2 + C)
        x1_minus = 0.5*(delta_b + 2*c2*x2_minus)/c1
        x2_pluss = B + np.sqrt(B**2 + C)
        x1_pluss = 0.5*(delta_b + 2*c2*x2_pluss)/c1

        if self._in_interval(x1_minus) and self._in_interval(x2_minus):
            x1 = x1_minus
            x2 = x2_minus
        elif self._in_interval(x1_pluss) and self._in_interval(x2_pluss):
            x1 = x1_pluss
            x2 = x2_pluss
        else:
            raise ValueError("Did not find any co-existence point!")

        return x1, x2

    def _in_interval(self, x):
        return x > 0.0 and x < 1.0

    def spinodal(self, phase):
        """
        Find the spinodal line
        """
        double_deriv = np.polyder(phase, m=2)
        roots = np.roots(double_deriv)
        return np.real(roots[~np.iscomplex(roots)])

    def plot(self, x, y, polys=[]):
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y)
        for p in polys:
            ax.plot(x, np.polyval(p, x))
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Free energy")

        y_range = np.max(y) - np.min(y)

        ax.set_ylim(np.min(y) - 0.05*y_range, np.max(y) + 0.05*y_range)
        return fig
