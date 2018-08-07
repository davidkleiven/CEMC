import numpy as np
from ase.units import kB
from scipy.integrate import cumtrapz


class CanonicalFreeEnergy(object):
    """
    Compute the Free Enery in the Canonical Ensemble (fixed composition)
    by thermodynamic integration

    :param composition: Dictionary with compositions
    """

    def __init__(self, composition, limit="hte", weights=None):
        self.comp = composition
        if isinstance(self.comp, dict):
            self.comp = [self.comp]  # Convert to a list with length 1
        self.limit = limit

        self.weights = weights
        if weights is None:
            self.weights = np.zeros(len(self.comp)) + 1.0/len(self.comp)

        if len(self.weights) != len(self.comp):
            msg = "The number of sublattices given compositions "
            msg += "and the number given in weights don't match!"
            raise ValueError(msg)

    def reference_energy(self, T, energy):
        """
        Computes the reference energy

        :param T: Temperature in kelvin
        :param energy: Internal energies (per atom)
        """
        if self.limit == "hte":
            infinite_temp_value = -self.inf_temperature_entropy()
            beta = 1.0 / (kB * T)
            ref_energy = infinite_temp_value + energy * beta
        else:
            beta = 1.0 / (kB * T)
            ref_energy = beta * energy
        return ref_energy

    def inf_temperature_entropy(self):
        """
        Return the entropy at infinite temperature
        """
        entropy_per_lattice = []
        for comp in self.comp:
            concs = np.array([value for key, value in self.comp.items()])
            infinite_temp_value = np.sum(concs * np.log(concs))
            entropy_per_lattice.append(infinite_temp_value)
        infinite_temp_value = self.weights.dot(entropy_per_lattice)
        return -infinite_temp_value

    def sort(self, temperature, internal_energy):
        """
        Sort the value such that the largerst temperature goes first

        :param temperature: Temperature in kelvin
        :param internal_energy: Internal energy (per atom)
        """
        if self.limit == "hte":
            srt_indx = np.argsort(temperature)[::-1]
        else:
            srt_indx = np.argsort(temperature)
        temp_srt = [temperature[indx] for indx in srt_indx]
        energy_srt = [internal_energy[indx] for indx in srt_indx]

        if self.limit == "hte":
            # Make sure the sorting is correct
            assert(temp_srt[0] > temp_srt[1])
        else:
            assert(temp_srt[0] < temp_srt[1])
        return np.array(temp_srt), np.array(energy_srt)

    def get(self, temperature, internal_energy):
        """
        Compute the Helholtz Free Energy

        :param temperature: Temperature in kelvin
        :param internal_energy: Internal energy (per atom)
        """
        temperature, internal_energy = self.sort(temperature, internal_energy)
        betas = 1.0 / (kB * temperature)
        ref_energy = self.reference_energy(temperature[0], internal_energy[0])
        free_energy = np.zeros(len(temperature))
        free_energy = cumtrapz(internal_energy, x=betas, initial=0.0)
        free_energy += ref_energy
        free_energy *= kB * temperature
        return temperature, internal_energy, free_energy
