import numpy as np


class BiasPotential(object):
    """
    Potential that can be used to manipulate the states visited.
    If added to MC run, the value returned from this will be
    added to the energy.
    """

    def __call__(self, system_changes):
        """Calculate the bias potential.

        :param list system_changes: List of tuples describing the changes
            see :py:method:`cemc.mcmc.mc_observers.MCObserver`
        """
        raise NotImplementedError("Has to be implemented in child classes!")

    def initialize(self):
        """Initialize the bias potential (if applicable).
           This functions is called right before MC sampling
           starts.
        """
        pass

    def calculate_from_scratch(self, atoms):
        """Calculate the bias potential from scratch.

        :param Atoms atoms: Structure to evaluate the bias poential for
        """
        raise NotImplementedError("Class has not implemented a calculate_from_scratch method!")

    def save(self, fname="pseudo_binary_free_energy.pkl"):
        """Save the computed bias potential to a file.

        :param str fname: Filename where a serialized version of this object
            will be stored.
        """
        import dill
        with open(fname, 'wb') as outfile:
            dill.dump(self, outfile)
        print("Pseudo binary free energy bias potential written to "
              "{}".format(fname))

    @staticmethod
    def load(fname="bias_potential.pkl"):
        """
        Load a bias potential from pickle file.
        Assume that this file has been stored with the save method
        of this class

        :param str fname: Filename of a serialized version of this object
        """
        import dill
        with open(fname, 'rb') as infile:
            obj = dill.load(infile)
        return obj


class SampledBiasPotential(BiasPotential):
    """
    Class for bias potentials that are sampled.

    :param reac_crd: Array with reaction coordinates
    :type reac_crd: list or numpy array
    :param free_eng: Array with energies
    :type free_eng: list or numpy array
    """

    def __init__(self, reac_crd=[], free_eng=[]):
        from scipy.interpolate import interp1d
        self.reac_crd = np.array(reac_crd)
        self.free_eng = np.array(free_eng)
        self.bias_interp = interp1d(self.reac_crd, self.free_eng,
                                    fill_value="extrapolate")

    def fit_smoothed_curve(self, smooth_length=11, show=False):
        """Fit a smoothed curve to the data.

        :param int smooth_length: Window length of the Savitzky-Golay filter.
            Has to be odd.
        :param bool show: If True, a plot of the resulting curve will be shown.
        """
        if smooth_length % 2 == 0:
            raise ValueError("smooth_length has to be an odd number!")
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(self.free_eng, smooth_length, 3)

        # Update the interpolator
        from scipy.interpolate import interp1d
        self.bias_interp = interp1d(self.reac_crd, smoothed,
                                    fill_value="extrapolate")

        if show:
            self.show()

    def show(self):
        """Create a plot of the bias potential.abs

        :return: Figure instance
        :rtype: Figure
        """
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.reac_crd, self.free_eng, 'o', mfc="none")
        ax.plot(self.reac_crd, self.bias_interp(self.reac_crd))
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Bias potential")
        return fig

    def __iadd__(self, other):
        """Add two bias potentials.

        :param BiasPotential other: The other potential

        :return: self
        :rtype: BiasPotential
        """
        from scipy.interpolate import interp1d
        self.free_eng += other.get(self.reac_crd)
        self.bias_interp = interp1d(self.reac_crd, self.free_eng,
                                    fill_value="extrapolate")
        return self

    def __add__(self, other):
        from copy import deepcopy
        new_obj = deepcopy(self)
        new_obj += other
        return new_obj


class PseudoBinaryFreeEnergyBias(SampledBiasPotential):
    """
    Class for a bias potential based on the free energy curve.
    It is intended for use when the reaction coordinate is the composition
    of one of the pseudobinary groups.
    NOTE: In the future this can in principle be extended to handle the
    case of arbitrary reaction paths

    :param pseudo_bin_conc_init: Instance of PseudoBinaryConcInitializer,
                                 can be None if __call__ is not called
                                 (i.e.) for fitting a smoothed curve etc.
    :type pseudo_bin_conc_init: PseudoBinaryConcInitializer or None
    :param reac_crd: Value of the reaction coordinate
    :type reac_crd: list or numpy array
    :param free_eng: Value of the free energy corresponding to the reac_crd
                     array
    :type free_eng: list or numpy array
    """

    def __init__(self, pseudo_bin_conc_init=None, reac_crd=[], free_eng=[]):
        self._conc_init = pseudo_bin_conc_init
        super(PseudoBinaryFreeEnergyBias, self).__init__(reac_crd, free_eng)

    @property
    def conc_init(self):
        from cemc.mcmc.reaction_path_utils import PseudoBinaryConcInitializer
        if not isinstance(self._conc_init, PseudoBinaryConcInitializer):
            raise TypeError("pseudo_bin_conc_init has to be of type "
                            "PseudoBinaryConcInitializer!")
        return self._conc_init

    @conc_init.setter
    def conc_init(self, init):
        from cemc.mcmc.reaction_path_utils import PseudoBinaryConcInitializer
        if not isinstance(init, PseudoBinaryConcInitializer):
            raise TypeError("pseudo_bin_conc_init has to be of type "
                            "PseudoBinaryConcInitializer!")
        self._conc_init = init

    def __call__(self, system_changes):
        """Evaluate the bias potential.

        see :py:method:`cemc.mcmc.mc_observers_MCObserver.__call__`
        """
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

    def calculate_from_scratch(self, atoms):
        """Calculate the value from scratch."""
        q = self.conc_init.get(atoms)
        return self.bias_interp(q)

    def get(self, reac_crd):
        """Get the bias potential as a function of reaction coordinate.

        :param float reac_crd: Reaction coordinate
        """
        return self.bias_interp(reac_crd)


class CovarianceBiasPotential(SampledBiasPotential):
    """
    Bias potential to be used together with
    :py:class:`cemc.mcmc.cov_reac_crd.CovarianceRangeConstraint`

    :param cov_range: Range constraints, mainly used to calculating the
        value of the reaction coordinate.
    :type cov_range: CovarianceRangeConstraint or None
    :param reac_crd: Reaction coordinates
    :type reac_crd: list or numpy array
    :param free_eng: Free energy at each reaction coordinate
    :type free_eng: list or numpy array
    """

    def __init__(self, cov_range=None, reac_crd=[], free_eng=[]):
        super(CovarianceBiasPotential, self).__init__(reac_crd, free_eng)
        self._cov_range = cov_range

    @property
    def cov_range(self):
        from cemc.mcmc import CovarianceRangeConstraint
        if not isinstance(self._cov_range, CovarianceRangeConstraint):
            raise TypeError("cov_range has to be of type "
                            "CovarianceRangeConstraint")
        return self._cov_range

    @cov_range.setter
    def cov_range(self, obj):
        self._cov_range = obj

    def __call__(self, system_changes):
        """Get the value for the bias potential.

        :param list system_changes: Proposed atom moves

        :return: Value of the bias potential after the move
        :rtype: float
        """
        reac_crd = self.cov_range.get_new_value(system_changes)
        return self.bias_interp(reac_crd)

    def calculate_from_scratch(self, atoms):
        """Calculate the value from scratch."""
        reac_crd = self.cov_range._cov_init.get(atoms, None)
        return self.bias_interp(reac_crd)

    def get(self, reac_crd):
        """Get the bias as a function reaction coordinate.abs

        :param float reac_crd: Reaction coordinate
        """
        return self.bias_interp(reac_crd)
