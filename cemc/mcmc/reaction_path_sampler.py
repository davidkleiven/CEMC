from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
import numpy as np
import time
import h5py as h5
import os


class ReactionPathSampler(object):
    """Class for sampling the free energy along some path."""

    def __init__(self, mc_obj=None, react_crd=[0.0, 1.0],
                 react_crd_init=None, react_crd_range_constraint=None,
                 n_windows=10, n_bins=10):
        self.mc = mc_obj
        self.react_crd = react_crd
        self.n_windows = n_windows
        self.n_bins = n_bins
        self.mpicomm = mc_obj.mpicomm
        mc_obj.mpicomm = None
        self.rank = 0
        if self.mpicomm is not None:
            self.rank = self.mpicomm.Get_rank()

        if not isinstance(react_crd_init, ReactionCrdInitializer):
            msg = "react_crd_init has to be of type "
            msg += "ReactionCrdInitializer"
            raise TypeError(msg)
        self.initializer = react_crd_init

        if not isinstance(react_crd_range_constraint,
                          ReactionCrdRangeConstraint):
            msg = "react_crd_range_constraint has to be of type "
            msg += "ReactionCrdRangeConstraint"
            raise TypeError(msg)
        self.constraint = react_crd_range_constraint
        self.mc.add_constraint(self.constraint)
        self.data = [np.ones(n_bins+1) for _ in range(n_windows)]
        self.window_range = (self.react_crd[1] - self.react_crd[0])/n_windows
        self.bin_range = self.window_range / n_bins

    def _get_window_limits(self, window):
        """
        Returns the upper and lower bound for window

        :param window: Index of window
        """
        maxval = self.react_crd[1]
        minval = self.react_crd[0]
        range_per_window = (maxval - minval) / self.n_windows
        min_lim = range_per_window * window

        # One bin is overlapping
        max_lim = range_per_window * (window + 1) + self.bin_range
        return min_lim, max_lim

    def _get_window_indx(self, window, value):
        """
        Returns the bin index of value in the current window

        :param window: Index of current window
        :param value: Value to be added in a histogram
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
        indx = (value - min_lim) / self.bin_range
        return int(indx)

    def _update_records(self):
        """
        Update the data arrays
        """
        reac_crd = self.initializer.get(self.mc.atoms)
        indx = self._get_window_indx(self.current_window, reac_crd)
        self.data[self.current_window][indx] += 1

    def _get_merged_records(self):
        """
        Merge the records into a one array
        """
        self._collect_results()
        all_data = self.data[0].tolist()
        for i in range(1, len(self.data)):
            ratio = float(all_data[-1]) / self.data[i][0]
            self.data[i] *= ratio
            all_data += self.data[i][1:].tolist()

        all_data = np.array(all_data)
        all_data /= all_data[0]
        G = -np.log(all_data)
        result = {}
        result["histogram"] = all_data
        result["free_energy"] = G
        result["x"] = np.linspace(self.react_crd[0], self.react_crd[1], len(G))
        return result

    def _bring_system_into_window(self):
        """Bring the system into the current window."""
        min, max = self._get_window_limits(self.current_window)
        val = 0.5 * (min + max)

        # We need to temporary release the concentration range
        # constraint when moving from a window
        self.mc.constraints.remove(self.constraint)

        # Bring the system into the new window
        self.initializer.set(self.mc.atoms, val)

        # Now add the constraint again
        self.mc.add_constraint(self.constraint)

    def log(self, msg):
        """Log messages."""
        if self.rank == 0:
            print(msg)

    def run(self, nsteps=10000):
        """
        Run MC simulation in all windows

        :param nsteps: Number of Monte Carlo step per window
        """

        output_every = 30
        # For all windows
        for i in range(self.n_windows):
            self.current_window = i
            # We are inside a new window, update to start with concentration in
            # the middle of this window
            min, max = self._get_window_limits(self.current_window)
            self.constraint.update_range([min, max])
            self._bring_system_into_window()
            # Now we are in the middle of the current window, start MC
            current_step = 0
            now = time.time()
            self.log("Initial chemical formula window {}: {}".format(
                self.current_window, self.mc.atoms.get_chemical_formula()))
            while (current_step < nsteps):
                current_step += 1
                if (time.time() - now > output_every):
                    self.log(
                        "Running MC step {} of {} in window {}".format(
                            current_step, nsteps, self.current_window))
                    now = time.time()

                # Run MC step
                self.mc._mc_step()
                self._update_records()

            self.log(
                "Acceptance rate in window {}: {}".format(
                    self.current_window, float(
                        self.mc.num_accepted) / self.mc.current_step))
            self.log("Final chemical formula: {}".format(
                self.mc.atoms.get_chemical_formula()))
            self.mc.reset()

    @staticmethod
    def dset_name(window):
        """Return a dataset name."""
        return "window{}".format(window)

    def _collect_results(self):
        """
        Collects the results from all processors
        """
        if (self.mpicomm is None):
            return
        temp_data = []
        for i in range(len(self.data)):
            recv_buf = np.zeros_like(self.data[i])
            self.mpicomm.Allreduce(self.data[i], recv_buf)
        self.data = temp_data

    def _update_data_entry(self, grp, data):
        """Update one data entry."""
        for key, value in data.items():
            try:
                old_data = np.array(grp[key])
                new_value = old_data + value
                entry = grp[key]
                entry[...] = new_value
            except KeyError:
                # The field does not exist
                grp.create_dataset(key, data=value)

    def save(self, fname="reaction_path.h5"):
        """Save data to HDF5 file."""
        res = self._get_merged_records()

        if self.rank == 0:
            # Write the histograms
            if os.path.exists(fname):
                flag = "r+"
            else:
                flag = "w"
            with h5.File(fname, flag) as hfile:
                for w in range(self.n_windows):
                    grp_name = ReactionPathSampler.dset_name(w)
                    try:
                        grp = hfile[grp_name]
                    except KeyError:
                        # The group does not exist
                        grp = hfile.create_group(grp_name)
                    data = {
                        "hist": self.data[w]
                    }
                    self._update_data_entry(grp, data)
                for key, value in res.items():
                    try:
                        data = hfile[key]
                        data[...] = value
                    except KeyError:
                        hfile.create_dataset(key, data=value)
