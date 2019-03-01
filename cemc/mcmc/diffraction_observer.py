from cemc.mcmc import MCObserver
from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
import numpy as np


class DiffractionUpdater(object):
    """
    Utility class for all objects that require tracing of a fourier
    reflection.

    :param Atoms atoms: Atoms object
    :param array k_vector: Fourier reflection to be traced
    :param list active_symbols: List of symbols that contributes to the
        reflection
    :param list all_symbols: List of all symbols in the simulation
    """
    def __init__(self, atoms=None, k_vector=[], active_symbols=[], 
                 all_symbols=[]):
        MCObserver.__init__(self)
        self.orig_symbols = [atom.symbol for atom in atoms]
        self.k_vector = k_vector
        self.N = len(atoms)
        self.k_dot_r = atoms.get_positions().dot(self.k_vector)
        self.indicator = {k: 0 for k in all_symbols}
        for symb in active_symbols:
            self.indicator[symb] = 1.0

        self.value = self.calculate_from_scratch(self.orig_symbols)
        self.prev_value = self.value

    def update(self, system_changes):
        """
        Update the reflection value
        """
        self.prev_value = self.value
        for change in system_changes:
            f_val = np.exp(1j*self.k_dot_r[change[0]])/self.N
            self.value += self.indicator[change[2]]*f_val
            self.value -= self.indicator[change[1]]*f_val
        return self.value

    def undo(self):
        """
        Undo the last update
        """
        self.value = self.prev_value

    def reset(self):
        """
        Reset all values
        """
        self.value = self.calculate_from_scratch(self.orig_symbols)
        self.prev_value = self.value

    def calculate_from_scratch(self, symbols):
        """Calculate the intensity from sctrach."""
        value = 0.0 + 1j*0.0
        for i, symb in enumerate(symbols):
            value += self.indicator[symb]*np.exp(1j*self.k_dot_r[i])
        self.value = value / len(symbols)
        return self.value


class DiffractionObserver(MCObserver):
    """
    Observer that traces the reflection intensity

    See docstring of :py:class:`cemc.mcmc.diffraction_observer.DiffractionUpdater`
    for explination of the arguments.
    """
    def __init__(self, atoms=None, k_vector=[], active_symbols=[],
                 all_symbols=[], name="reflect"):
        MCObserver.__init__(self)
        self.updater = DiffractionUpdater(
            atoms=atoms, k_vector=k_vector, active_symbols=active_symbols,
            all_symbols=all_symbols)
        self.avg = self.updater.value
        self.num_updates = 1
        self.name = name

    def __call__(self, system_changes, peak=False):
        self.updater.update(system_changes)
        if peak:
            cur_val = self.get_current_value()
            self.updater.undo()
        else:
            self.avg += self.updater.value
            cur_val = self.get_current_value()
        return cur_val

    def get_averages(self):
        return {self.name: np.abs(self.avg/self.num_updates)}

    def reset(self):
        self.updater.reset()
        self.avg = self.updater.value
        self.num_updates = 1

    def get_current_value(self):
        return {self.name: np.abs(self.updater.value)}


class DiffractionRangeConstraint(ReactionCrdRangeConstraint):
    """
    Constraints based on diffraction intensity

    See docstring of :py:class:`cemc.mcmc.diffraction_observer.DiffractionUpdater`
    for explination of the arguments.
    """
    def __init__(self, updater):

        ReactionCrdRangeConstraint.__init__(self)
        self.updater = updater

    def __call__(self, system_changes):
        """
        Check if the constraint is violated after update.
        """
        new_val = np.abs(self.updater.update(system_changes))
        self.updater.undo()
        return new_val >= self.range[0] and new_val < self.range[1]

    def update(self, system_changes):
        self.updater.update(system_changes)


class DiffractionCrdInitializer(ReactionCrdInitializer):
    """
    Diffraction coordinate.

    See docstring of :py:class:`cemc.mcmc.diffraction_observer.DiffractionUpdater`
    for explination of the arguments.
    """
    def __init__(self, updater):
        ReactionCrdInitializer.__init__(self)
        self.updater = updater

    def get(self, atoms, system_changes=[]):
        """
        Get the value of the current reflection intensity
        """
        if system_changes:
            value = np.abs(self.updater.update(system_changes))
            self.updater.undo()
            return value
        elif atoms is not None:
            symbols = [atom.symbol for atom in atoms]
            # NOTE: This updates the value!
            self.updater.calculate_from_scratch(symbols)
        return np.abs(self.updater.value)

    def update(self, system_changes):
        self.updater.update(system_changes)
