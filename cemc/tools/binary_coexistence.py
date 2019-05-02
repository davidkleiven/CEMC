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

        x2 = B - np.sqrt(B**2 + C)
        x1 = 0.5*(delta_b + 2*c2*x2)/c1
        return x1, x2

    def spinodal(self, phase):
        """
        Find the spinodal line
        """
        double_deriv = np.polyder(phase, m=2)
        return np.roots(double_deriv)
