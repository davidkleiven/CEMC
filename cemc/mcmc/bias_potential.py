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
        from scipy.interpolate import interp1d
        self._conc_init = pseudo_bin_conc_init
        self.reac_crd = reac_crd
        self.free_eng = free_eng
        self.bias_interp = interp1d(self.reac_crd, self.free_eng,
                                    fill_value="extrapolate")

    def fit_smoothed_curve(self, smooth_length=11, show=False):
        """Fit a smoothed curve to the data."""
        if smooth_length % 2 == 0:
            raise ValueError("smooth_length has to be an odd number!")
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(self.free_eng, smooth_length, 3)

        # Update the interpolator
        from scipy.interpolate import interp1d
        self.bias_interp = interp1d(self.reac_crd, smoothed,
                                    fill_value="extrapolate")

        if show:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.reac_crd, self.free_eng, 'o', mfc="none")
            ax.plot(self.reac_crd, self.bias_interp(self.reac_crd))
            ax.set_xlabel("Reaction coordinate")
            ax.set_ylabel("Bias potential")
            plt.show()

    @property
    def conc_init(self):
        from cemc.mcmc import PseudoBinaryConcInitializer
        if not isinstance(self._conc_init, PseudoBinaryConcInitializer):
            raise TypeError("pseudo_bin_conc_init has to be of type "
                            "PseudoBinaryConcInitializer!")
        return self._conc_init

    @conc_init.setter
    def conc_init(self, init):
        from cemc.mcmc import PseudoBinaryConcInitializer
        if not isinstance(init, PseudoBinaryConcInitializer):
            raise TypeError("pseudo_bin_conc_init has to be of type "
                            "PseudoBinaryConcInitializer!")
        self._conc_init = init

    def save(self, fname="pseudo_binary_free_energy.pkl"):
        """Save the computed bias potential to a file."""
        import pickle
        with open(fname, 'wb') as outfile:
            pickle.dump(self, outfile)
        print("Pseudo binary free energy bias potential written to "
              "{}".format(fname))

    @staticmethod
    def load(fname="pseudo_binary_free_energy.pkl"):
        """
        Load a bias potential from pickle file.
        Assume that this file has been stored with the save method
        of this class
        """
        import pickle
        with open(fname, 'rb') as infile:
            obj = pickle.load(infile)
        return obj

    def __call__(self, system_changes):
        """Evaluate the bias potential."""
        # We can pass None  in this case
        # PseudoBinaryConcInitializer should track the atoms object itself
        q = self.conc_init.get(None)
        symb = self.conc_init.target_symb
        for change in system_changes:
            if change[1] == symb:
                q -= 1.0 / self.conc_init.num_per_unit
            elif change[2] == symb:
                q += 1.0 / self.conc_init.num_per_unit
        return self.bias_interp(q)

    def get(self, reac_crd):
        """Get the bias potential as a function of reaction coordinate."""
        return self.bias_interp(reac_crd)
