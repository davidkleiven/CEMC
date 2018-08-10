import numpy as np
from ase.units import kB
import copy


class ExponentialWeightedAverager(object):
    """Class for averaging boltzmann sum
    sum_i E_i^order exp(-E_i/kT)

    :param temp: Temperature
    :param order: Order of the value to sample
    """

    def __init__(self, temp, order=0):
        self.T = temp
        self.ref_value = 0.0
        self._average = 0.0
        self._num_samples = 0
        self.beta = 1.0/(kB*temp)
        self.order = order

    def add(self, value):
        """Add a new value to the sampler."""
        if value < self.ref_value:
            diff = value - self.ref_value
            self._average *= np.exp(self.beta * diff)
            self._average += value**self.order
            self.ref_value = value
        else:
            diff = value - self.ref_value
            self._average += np.exp(-self.beta * diff) * value**self.order
        self._num_samples += 1

    @property
    def average(self):
        return self._average/self._num_samples

    def reset(self):
        """Reset the reference function."""
        self.ref_value = 0.0
        self._average = 0.0
        self._num_samples = 0

    def __iadd__(self, other):
        """Add two averaging objects."""
        if abs(self.T - other.T) > 1E-5:
            msg = "The two objects being added needs to have the same "
            msg += "temperature."
            raise ValueError(msg)

        if self.ref_value < other.ref_value:
            diff = self.ref_value - other.ref_value
            other._average *= np.exp(self.beta * diff)
            other.ref_value = self.ref_value
        else:
            diff = other.ref_value - self.ref_value
            self.ref_value = other.ref_value
            self._average *= np.exp(self.beta * diff)
        self._average += other._average
        self._num_samples += other._num_samples
        return self

    def __add__(self, other):
        """Addition operator."""
        obj = copy.deepcopy(self)
        obj += other
        return obj
