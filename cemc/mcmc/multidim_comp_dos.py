import numpy as np
from scipy.interpolate import UnivariateSpline


class CompositionDOS(object):
    def __init__(self, conc, hist):
        self.dim = len(hist.shape)
        if self.dim > 1:
            msg = "Currently only up to 1 dimensional arrays are supported"
            raise ValueError(msg)
        self.hist = hist

        if self.dim == 1:
            self.conc = np.linspace(conc[0][0], conc[0][1], len(self.hist))
            self.conc = self.conc[self.hist > 0]
            self.hist = self.hist[self.hist > 0]
            self.interpolator = UnivariateSpline(self.conc, self.hist)

    def __call__(self, conc):
        return self.interpolator(conc)

    def plot1D(self):
        """Plot free energy as a function of composition."""
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.conc, self.hist, ls="steps")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Free Energy")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return ax

    def plot(self):
        """Plot free energy as a function of composition."""
        if self.dim == 1:
            return self.plot1D()
