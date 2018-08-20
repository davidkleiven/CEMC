class BiasPotential(object):
    """
    Potential that can be used to manipulate the states visited.
    If added to MC run, the value returned from this will be
    added to the energy.
    """

    def __call__(self, system_changes):
        raise NotImplementedError("Has to be implemented in child classes!")


class PseudoBinaryFreeEnergyBias(BiasPotential):
    def __init__(self, pseudo_bin_conc_init=None, reac_crd=[], free_eng=[]):
        from cemc.mcmc import PseudoBinaryConcInitializer
        from scipy.interpolate import interp1d
        if not isinstance(pseudo_bin_conc_init, PseudoBinaryConcInitializer):
            raise TypeError("pseudo_bin_conc_init has to be of type "
                            "PseudoBinaryConcInitializer!")
        self.conc_init = pseudo_bin_conc_init
        self.reac_crd = reac_crd
        self.free_eng = free_eng
        self.bias_intep = interp1d(self.reac_crd, self.free_eng,
                                   fill_value="extrapolate")

    def fit_smoothed_curve(self, smooth_length=11, show=False):
        """Fit a smoothed curve to the data."""
        if smooth_length % 2 == 0:
            raise ValueError("smooth_length has to be an odd number!")
        from scipy.signal import savgol_filter
        self.free_eng = savgol_filter(self.free_eng, smooth_length, 3)

        if show:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.reac_crd, self.free_eng)
            ax.set_xlabel("Reaction coordinate")
            ax.set_ylabel("Bias potential")
            plt.show()

        # Update the interpolator
        from scipy.interpolate import interp1d
        self.bias_intep = interp1d(self.reac_crd, self.free_eng,
                                   fill_value="extrapolate")

    def __call__(self, system_changes):
        """Evaluate the bias potential."""
        # We can pass None  in this case
        # PseudoBinaryConcInitializer should track the atoms object itself
        q = self.conc_init.get(None)
        return self.bias_interp(q)
