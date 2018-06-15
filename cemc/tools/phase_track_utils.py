import numpy as np
from cemc.wanglandau.ce_calculator import CE

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
