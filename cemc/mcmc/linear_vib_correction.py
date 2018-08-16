from ase.units import kB
import copy
import numpy as np


class LinearVibCorrection(object):
    def __init__(self, eci_per_kbT):
        self.eci_per_kbT = eci_per_kbT
        self.orig_eci = None
        self.current_eci = None
        self.vibs_included = False
        self.temperature = 0.0

    def check_provided_eci_match(self, provided_eci):
        """
        Check that the provided ecis match the ones stored in this object
        """
        if self.current_eci is None:
            return
        return
        for key in self.current_eci.keys():
            if key not in provided_eci.keys():
                msg = "The keys don't match! "
                msg += "Keys should be: {}. ".format(self.current_eci.keys())
                msg += "Keys provided: {}".format(provided_eci.keys())
                raise ValueError(msg)

        for key in self.current_eci.keys():
            if not np.isclose(self.current_eci[key], provided_eci[key]):
                msg = "Provided ECIs do not match the ones stored. "
                msg += "Stored: {}. ".format(self.current_eci)
                msg += "Provided: {}".format(provided_eci)
                raise ValueError(msg)

    def include(self, eci_with_out_vib, T):
        """Add the vibrational ECIs to the orignal."""
        self.temperature = T
        if self.vibs_included:
            return
        self.check_provided_eci_match(eci_with_out_vib)
        for key in self.eci_per_kbT.keys():
            if key not in eci_with_out_vib.keys():
                msg = "The cluster {} is not in the original ECIs!".format(key)
                raise KeyError(msg)

        for key, value in self.eci_per_kbT.items():
            eci_with_out_vib[key] += value * kB * T
        self.vibs_included = True
        self.current_eci = copy.deepcopy(eci_with_out_vib)
        return eci_with_out_vib

    def reset(self, eci_with_vibs):
        """Remove the contribution from vibrations to the ECIs."""
        self.check_provided_eci_match(eci_with_vibs)
        if not self.vibs_included:
            return
        for key, value in self.eci_per_kbT.items():
            eci_with_vibs[key] -= value * kB * self.temperature
        self.current_eci = copy.deepcopy(eci_with_vibs)
        self.vibs_included = False
        return eci_with_vibs

    def energy_due_to_vibrations(self, T, cf):
        """
        Computes the contribution to the energy due to internal vibrations
        """
        E = 0.0
        for key, value in self.eci_per_kbT:
            E += value * cf[key] * kB * T
        return E
