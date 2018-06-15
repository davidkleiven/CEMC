import numpy as np
import h5py as h5

from cemc.wanglandau.ce_calculator import CE
from cemc.tools import DatasetAverager

class PhaseBoundarySolution(object):
    """
    Class for tracking the solution of the phase boundary tracing
    """
    def __init__(self):
        self.singlets = []
        self.temperatures = []
        self.chem_pot = []

    def append(self, singlet, temp, chem_pot):
        """
        Append a new entry

        :param singlet: New singlet entry
        :param temp: New temperature
        :param chem_pot: New chemical potantial
        """
        self.singlets.append(np.copy(singlet))
        self.temperatures.append(temp)
        self.chem_pot.append(np.copy(chem_pot))

    def to_dict(self):
        """
        Return a dictionary with the elements
        """
        dictionary = {}
        dictionary["singlets"] = self.singlets
        dictionary["temperatures"] = self.temperatures
        dictionary["chem_pot"] = self.chem_pot
        return dictionary

class CECalculators(object):
    """
    Class that manages the cluter expansion calculators as well as their
    initial state
    """
    def __init__(self, ground_states):
        self.calcs = []
        self.orig_symbols = []
        for ground_state in ground_states:
            self.calcs.append(CE(ground_state["bc"], ground_state["eci"], initial_cf=ground_state["cf"]))
            ground_state["bc"].atoms.set_calculator(self.calcs[-1])
            symbs = [atom.symbol for atom in ground_state["bc"].atoms]
            self.orig_symbols.append(symbs)

    def reset(self):
        """
        Rests all the calculators to their original state
        """
        for calc, symbs in zip(self.calcs, self.orig_symbols):
            calc.set_symbols(symbs)

def save_phase_boundary(fname, result):
    """
    Saves the phase boundary to a h5 file. If exists, it is appended.
    """
    with h5.File(fname, 'a') as hfile:
        names = hfile.keys()
        new_name = "boundary{}".format(len(names)+1)
        grp = hfile.create_group(new_name)
        grp.attrs["is_phase_boundary"] = True
        for key, value in result.items():
            dset = grp.create_dataset(key, data=value)
    print("Result written to {}".format(fname))

def process_phase_boundary(fname):
    """
    Processes the phase boundary file, computed mean and standard deviations
    """
    from scipy.interpolate import interp1d
    singlets = []
    chem_pot = []
    temperatures = []
    with h5.File(fname, 'r') as hfile:
        for name in hfile.keys():
            grp = hfile[name]
            singlets.append(np.array(grp["singlets"]))
            chem_pot.append(np.array(grp["chem_pot"]))
            temperatures.append(np.array(grp["temperatures"]))

    max_temp = 0.0
    min_temp = 10000000.0
    for temp_array in temperatures:
        if np.max(temp_array) > max_temp:
            max_temp = np.max(temp_array)
        if np.min(temp_array) < min_temp:
            min_temp = np.min(temp_array)

    temp_linspace = np.linspace(min_temp, max_temp, 200)
    result = {}
    result["chem_pot"] = []
    result["std_chem_pot"] = []
    result["singlets"] = []
    result["std_singlets"] = []
    result["num_visits"] = []
    result["temperature"] = temp_linspace

    for sing_dset in singlets:
        if np.any(sing_dset.shape != singlets[0].shape):
            msg = "Invalid file! Looks like it contains phase boundary\n"
            msg += " data for different systems"
            raise ValueError(msg)

    num_chem_pots = chem_pot[0].shape[1]
    for i in range(num_chem_pots):
        mu_averager = DatasetAverager(temp_linspace)
        for temps, mu in zip(temperatures, chem_pot):
            mu_averager.add_dataset(temps, mu[:,i])
        mu_res = mu_averager.get()
        result["chem_pot"].append(mu_res["y_values"])
        result["std_chem_pot"].append(mu_res["std_y"])
        result["num_visits"].append(mu_res["num_visits"])

    num_singlets = singlets[0].shape[1]
    for i in range(num_chem_pots):
        for temp, singl in zip(temperatures, singlets):
            singlet_averager = DatasetAverager(temp_linspace)
            singlet = []
            std_singlet = []
            for j in range(num_singlets):
                singlet_averager.add_dataset(temps, singl[:,j,i])
            singl_res = singlet_averager.get()
            singlet.append(singl_res["y_values"])
            std_singlet.append(singl_res["std_y"])
        result["singlets"].append(singlet)
        result["std_singlets"].append(std_singlet)
    return result
