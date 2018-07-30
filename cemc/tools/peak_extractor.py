import numpy as np


class PeakExtractor(object):
    """Extract peaks from xy-datasets.

    :param x: x-values
    :param y: y-values
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _locate_peak(self, xfit, yfit):
        """Locate peak by fitting a parabola."""
        coeff = np.polyfit(xfit, yfit, 2)
        x_peak = -0.5*coeff[1]/coeff[0]
        y_peak = coeff[0] * x_peak**2 + coeff[1] * x_peak + coeff[2]
        return x_peak, y_peak

    def peaks(self):
        """Locate peaks.

        :return: List of dictionaries
                 [{"x": x_peak1, "y": y_peak1, "indx1": 3},
                 {"x": x_peak2, "y": y_peak2, "indx2": 8}]
        """

        peaks = []
        for i in range(1, len(self.x)-1):
            if self.y[i] > self.y[i-1] and self.y[i] > self.y[i+1]:
                # We have a peak
                x_peak, y_peak = self._locate_peak(self.x[i-1: i+2],
                                                   self.y[i-1: i+2])
                peaks.append({"x": x_peak, "y": y_peak, "indx": i})
        return peaks
