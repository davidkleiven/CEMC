from cemc.mcmc import SGCMonteCarlo
import numpy as np
from ase.io import write
import json
import time


class SGCFreeEnergyBarrier(SGCMonteCarlo):
    """
    Class for computing the free energy as a function of composition
    It applies umbrella sampling to force the system to also visit
    regions with low statistical weight

    :param Atoms atoms: Atoms object
    :param float T: Temperature in kelvin
    :param int n_windows: Number of windows used for Umbrellas sampling
        with hard wall bias potential
    :param int n_bins: Number of bins within each window
    :param float min_singlet: Minimum value of the singlet value
    :param float max_singlet: Maximum value of the singlet value
    :param other: See :py:class:`cemc.mcmc.SGCMonteCarlo`
    """

    def __init__(self, atoms, T, **kwargs):
        n_windows = 60
        n_bins = 5
        min_singlet = -1.0
        max_singlet = -0.4
        if "n_windows" in kwargs.keys():
            n_windows = kwargs.pop("n_windows")
        if "n_bins" in kwargs.keys():
            n_bins = kwargs.pop("n_bins")
        if "min_singlet" in kwargs.keys():
            min_singlet = kwargs.pop("min_singlet")
        if "max_singlet" in kwargs.keys():
            max_singlet = kwargs.pop("max_singlet")

        self.recycle_waste = False
        if "recycle_waste" in kwargs.keys():
            self.recycle_waste = kwargs.pop("recycle_wast")

        self.save_last_state_in_window = False
        if "save_last_state" in kwargs.keys():
            self.save_last_state_in_window = kwargs.pop("save_last_state")

        self.n_windows = n_windows
        self.n_bins = n_bins
        self.min_singlet = min_singlet
        self.max_singlet = max_singlet
        self.kwargs_parent = kwargs
        super(SGCFreeEnergyBarrier, self).__init__(atoms, T, **kwargs)
        self.chem_potential_restart_file = None

        # Set up whats needed
        self.window_singletrange = (
            self.max_singlet - self.min_singlet) / self.n_windows
        self.bin_singletrange = self.window_singletrange / self.n_bins
        self.energydata = []
        self.current_window = 0
        self.data = []
        self.result = {}
        for i in range(self.n_windows):
            if (i == 0):
                self.data.append(np.ones(self.n_bins))
                self.energydata.append(np.zeros(self.n_bins))
            else:
                self.data.append(np.ones(self.n_bins + 1))
                self.energydata.append(np.zeros(self.n_bins + 1))

    def _get_window_limits(self, window):
        """
        Returns the upper and lower bound for window

        :param int window: Index of window

        :return: Lower and upper bound of the window
        :rtype: float, float
        """
        if window == 0:
            min_limit = self.min_singlet
        else:
            min_limit = window * self.window_singletrange - self.bin_singletrange

        min_limit = self.min_singlet + abs(self.window_singletrange) * window
        max_limit = (window + 1) * self.window_singletrange
        max_limit = self.min_singlet + \
            abs(self.window_singletrange) * (window + 1)
        return min_limit, max_limit

    def _get_window_indx(self, window, value):
        """
        Returns the bin index of value in the current window

        :param int window: Index of current window
        :param float value: Value to be added in a histogram

        :return: Bin in the current window of the value passed
        :rtype: int
        """
        min_lim, max_lim = self._get_window_limits(window)
        if (value < min_lim or value >= max_lim):
            msg = "Value out of range for window\n"
            msg += "Value has to be in range [{},{})\n".format(
                min_lim, max_lim)
            msg += "Got value: {}".format(value)
            raise ValueError(msg)

        if (window == 0):
            N = self.n_bins
        else:
            N = self.n_bins + 1
        indx = (value - min_lim) * N / (max_lim - min_lim)
        return int(indx)

    def _update_records(self):
        """
        Update the data arrays
        """
        singlet = self.averager.singlets[0]
        indx = self._get_window_indx(self.current_window, singlet)
        self.data[self.current_window][indx] += 1
        self.energydata[self.current_window][indx] += self.current_energy

    def _accept(self, system_changes):
        """
        Return True if the trial move was accepted, False otherwise

        :param list system_changes: List of system changes

        :return: True/False if the move is accepted or not
        :rtype: bool
        """

        # Check if move accepted by SGCMonteCarlo
        move_accepted = SGCMonteCarlo._accept(self, system_changes)

        # Now check if the move keeps us in same window
        new_singlets = self.atoms._calc.get_singlets()
        singlet = new_singlets[0]
        # Set in_window to True and check if it should be False instead
        in_window = True
        min_allowed, max_allowed = self._get_window_limits(self.current_window)

        if (singlet < min_allowed or singlet >= max_allowed):
            in_window = False
        # Now system will return to previous state if not inside window
        return move_accepted and in_window

    def _is_inside_window(self, singlet):
        """Check if the current state is inside the window.

        :param float singlet: Singlet value

        :return: True/False. If True, the given value is inside the window
        :rtype: bool
        """
        min_allowed, max_allowed = self._get_window_limits(self.current_window)
        return singlet >= min_allowed and singlet < max_allowed

    def _get_merged_records(self):
        """
        Merge the records into a one array

        :return: Merged results (keys: histogram, free_energy, energy)
        :rtype: dict
        """
        all_data = self.data[0].tolist()
        energy = (self.energydata[0] / self.data[0]).tolist()
        for i in range(1, len(self.data)):
            ratio = all_data[-1] / self.data[i][0]
            self.data[i] *= ratio
            all_data += self.data[i][1:].tolist()
            energy += (self.energydata[i] / self.data[i])[1:].tolist()

        all_data = np.array(all_data)
        all_data /= all_data[0]
        G = -np.log(all_data)
        self.result["histogram"] = all_data.tolist()
        self.result["free_energy"] = G.tolist()
        self.result["energy"] = energy
        return self.result

    def save(self, fname="sgc_free_energy.json"):
        """
        Stores the results to a JSON file

        :param str fname: Filename
        """
        self._get_merged_records()

        self.result["temperature"] = self.T
        self.result["chemical_potential"] = self.chemical_potential
        x = np.linspace(
            self.min_singlet, self.max_singlet, len(
                self.result["free_energy"]))
        self.result["xaxis"] = x.tolist()
        self.result["min_singlet"] = self.min_singlet
        self.result["max_singlet"] = self.max_singlet
        self.result["kwargs_parent"] = self.kwargs_parent

        # Store also the raw histgramgs
        self.result["raw_histograms"] = [hist.tolist() for hist in self.data]
        self.result["raw_energydata"] = [value.tolist() for value in self.energydata]

        with open(fname, 'w') as outfile:
            json.dump(
                self.result,
                outfile,
                indent=2,
                separators=(
                    ",",
                    ":"))
        self.log("Results written to: {}".format(fname))

    @staticmethod
    def load(atoms, fname):
        """
        Loads the results from a file such the calculations can be restarted
        Returns an instance of the object in the same state as it was left

        :param str fname: Filename to data file
        """
        with open(fname, 'r') as infile:
            params = json.load(infile)
        T = params["temperature"]
        chem_potential_restart_file = params["chemical_potential"]
        data = [np.array(hist) for hist in params["raw_histograms"]]
        energydata = params["raw_energydata"]
        n_windows = len(data)
        n_bins = len(data[0])
        min_singlet = params["min_singlet"]
        max_singlet = params["max_singlet"]
        kwargs_parent = params["kwargs_parent"]
        obj = SGCFreeEnergyBarrier(
            atoms,
            T,
            n_windows=n_windows,
            n_bins=n_bins,
            min_singlet=min_singlet,
            max_singlet=max_singlet,
            **kwargs_parent)

        obj.chem_potential_restart_file = chem_potential_restart_file

        # Insert the data into the data array
        for i in range(len(data)):
            obj.data[i][:] = np.array(data[i])
            obj.energydata[i][:] = np.array(energydata[i])
        return obj

    def _bring_system_into_window(self):
        """
        Brings the system into the current window
        """
        min_lim, max_lim = self._get_window_limits(self.current_window)
        newsinglet = 0.5 * (min_lim + max_lim)
        self.atoms._calc.set_singlets({"c1_0": newsinglet})

    def run(self, nsteps=10000, chem_pot=None):
        """
        Run MC simulation in all windows

        :param int nsteps: Number of Monte Carlo step per window
        :param dict chem_pot: Chemical potential.
            See :py:meth:`cemc.mcmc.SGCMonteCarlo.runMC`
        """
        if (self.chem_potential_restart_file is not None):
            self.log("Chemical potential was read from the restart file.")
            self.chemical_potential = self.chem_potential_restart_file
        else:
            self.chemical_potential = chem_pot

        output_every = 30
        # For all windows
        for i in range(self.n_windows):
            self.current_window = i
            # We are inside a new window, update to start with concentration in
            # the middle of this window
            self._bring_system_into_window()
            self.is_first = True
            # Now we are in the middle of the current window, start MC
            current_step = 0
            now = time.time()
            self.log("Initial chemical formula window {}: {}".format(
                self.current_window, self.atoms.get_chemical_formula()))
            while (current_step < nsteps):
                current_step += 1
                if (time.time() - now > output_every):
                    self.log(
                        "Running MC step {} of {} in window {}".format(
                            current_step, nsteps, self.current_window))
                    now = time.time()

                # Run MC step
                self._mc_step()
                self._update_records()
                self.averager.reset()

            self.log(
                "Acceptance rate in window {}: {}".format(
                    self.current_window, float(
                        self.num_accepted) / self.current_step))
            self.log("Final chemical formula: {}".format(
                self.atoms.get_chemical_formula()))
            self.reset()

            if self.save_last_state_in_window:
                fname = "last_state_in_{}_{}K.xyz".format(i, self.T)
                write(fname, self.atoms)
                self.log("Structure written to {}".format(fname))
        self._get_merged_records()

    @staticmethod
    def plot(fname="sgc_data.json"):
        """
        Create some standard plots of the results file produced

        :param str fname: Filename of the data file

        :return: All figures
        :rtype: list of Figure
        """
        from matplotlib import pyplot as plt
        with open(fname, 'r') as infile:
            result = json.load(infile)
        print("Temperature: {}K".format(result["temperature"]))
        print("Chemical potential: {}K".format(result["chemical_potential"]))

        figs = []
        figs.append(plt.figure())
        ax = figs[-1].add_subplot(1, 1, 1)
        ax.plot(result["xaxis"], result["free_energy"], ls="steps")
        ax.set_xlabel("Singlets")
        ax.set_ylabel("$\\beta \\Delta G$")

        figs.append(plt.figure())
        ax = figs[-1].add_subplot(1, 1, 1)
        ax.plot(result["xaxis"], result["energy"], ls="steps")
        ax.set_xlabel("Singlets")
        ax.set_ylabel("Energy (eV)")
        return figs
