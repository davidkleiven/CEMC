from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
import numpy as np
import time
import h5py as h5
import os
import sys


class ReactionPathSampler(object):
    """Generic class for sampling the free energy along some path.

    :param Montecarlo mc_obj: MC object
    :param list reac_crd: Maximum and minimum limits of the range of the
        reaction coordinate. [lower, upper)
    :param ReactionCrdInitializer react_crd_init: Has to be able to generate
        configurations corresponding to an arbitrary reaction coordinate.
    :param ReactionCrdRangeConstraint react_crd_range_constraint: Given a set
        of changes it has to return True if the object is still inside the
        range, and False otherwise
    :param int n_window: Number of windows used for Umbrella sampling
    :param int n_bins: Number of bins inside each window
    :param str data_file: HDF5-file where all data acquired during the run
        is stored
    """

    def __init__(self, mc_obj=None, react_crd=[0.0, 1.0],
                 react_crd_init=None, react_crd_range_constraint=None,
                 n_windows=10, n_bins=10, data_file="reaction_path.h5"):
        self.mc = mc_obj
        self.react_crd = react_crd
        self.n_windows = n_windows
        self.n_bins = n_bins
        self.mpicomm = mc_obj.mpicomm
        mc_obj.mpicomm = None
        self.rank = 0
        mc_obj.rank = 0  # All MC object should think they have rank 0
        self.fname = data_file
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

        :param int window: Index of window

        :return: Lower and upper limit of the window
        :rtype: float, float
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

        :param int window: Index of current window
        :param float value: Value to be added in a histogram

        :return: Bin in the window corresponding to the value given
        :rtype: int
        """
        min_lim, max_lim = self._get_window_limits(window)
        if (value < min_lim or value >= max_lim):
            msg = "Value out of range for window\n"
            msg += "Value has to be in range [{},{})\n".format(
                min_lim, max_lim)
            msg += "Got value: {}".format(value)
            raise ValueError(msg)

        N = self.n_bins + 1
        indx = (value - min_lim) * N / (max_lim - min_lim)
        indx = (value - min_lim) / self.bin_range
        return int(indx)

    def _update_records(self):
        """
        Update the data arrays
        """
        try:
            reac_crd = self.initializer.get(self.mc.atoms)
            indx = self._get_window_indx(self.current_window, reac_crd)

            # Add a safety in case it is marginally larger
            if indx >= len(self.data[self.current_window]):
                indx -= 1
            self.data[self.current_window][indx] += 1
        except Exception as exc:
            print("Error on rank: {}".self.rank)
            print("{}: {}".format(type(exc).__name__, str(exc)))
            print("Errors here will not be handled, because it will lead "
                  "to tremendous communication overhead.")

    def _get_merged_records(self, data):
        """
        Merge the records into a one array

        :param list data: List of histograms containing the number of visits in
            each window

        :return: Merged values. Keys: histrogram, free_energy, x
        :rtype: dict
        """
        all_data = data[0].tolist()
        for i in range(1, len(data)):
            ratio = float(all_data[-1]) / data[i][0]
            all_data += (data[i][1:] * ratio).tolist()

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
        error = 0
        msg = ""
        try:
            # Bring the system into the new window
            self.initializer.set(self.mc.atoms, val)
        except Exception as exc:
            # Indicate that an error occured. It will be handled by
            # another function
            msg = "{}: {}".format(type(exc).__name__, str(exc))
            error = 1

        if self.mpicomm is not None:
            error = self.mpicomm.allreduce(error)

        # Raise error on all processes
        if error:
            raise Exception(msg)

        # Now add the constraint again
        self.mc.add_constraint(self.constraint)

    def log(self, msg, ranks=[0]):
        """Log messages.

        :param str msg: Message to log
        :param list ranks: Ranks that should print the message. Default is
            only the master process.
        """
        if self.rank in ranks:
            print(msg)
            sys.stdout.flush()

    def log_window_statistics(self, window):
        """Print logging message concerning the sampling window.

        :param int window: Int, window to log statistics from.
        """
        max = np.max(self.data[window])
        min = np.min(self.data[window])
        mean = np.mean(self.data[window])
        msg = "Window: {}. Min: {} Max: {} ".format(window, min, max)
        msg += "Mean: {}".format(mean)
        self.log(msg)

    def save_current_window(self):
        """Save result of current window to file."""
        self.log("Collecting results from all processes...")
        self._collect_results(self.current_window)
        if os.path.exists(self.fname):
            flag = "r+"
        else:
            flag = "w"

        if self.rank == 0:
            with h5.File(self.fname, flag) as hfile:
                grp_name = ReactionPathSampler.dset_name(self.current_window)
                try:
                    grp = hfile[grp_name]
                except KeyError:
                    # The group does not exist
                    grp = hfile.create_group(grp_name)
                data = {
                    "hist": self.data[self.current_window]
                }
                self._update_data_entry(grp, data)

    @property
    def converged_all_bins(self):
        """Check if all bins have been covered."""
        nproc = 1
        if self.mpicomm is not None:
            nproc = self.mpicomm.Get_size()
        for dset in self.data:
            if np.min(dset) < nproc + 0.1:
                return False
        return True

    def run(self, nsteps=10000):
        """
        Run MC simulation in all windows

        :param int nsteps: Number of Monte Carlo step per window
        """
        from cemc.mcmc import CanNotFindLegalMoveError
        output_every = 30
        # For all windows
        for i in range(self.n_windows):
            self.current_window = i
            # We are inside a new window, update to start with concentration in
            # the middle of this window
            min, max = self._get_window_limits(self.current_window)
            self.constraint.update_range([min, max])
            self.log("Bringing system into window...")
            self._bring_system_into_window()

            # Now we are in the middle of the current window, start MC
            current_step = 0
            now = time.time()
            self.log("Initial chemical formula window {}: {}".format(
                self.current_window, self.mc.atoms.get_chemical_formula()))
            print("Starting MC calculation on rank: {}".format(self.rank))
            num_failed_attempts = 0
            while (current_step < nsteps):
                current_step += 1
                if (time.time() - now > output_every):
                    self.log(
                        "Running MC step {} of {} in window {}".format(
                            current_step, nsteps, self.current_window))
                    now = time.time()

                # Run MC step
                try:
                    self.mc._mc_step()
                    self._update_records()
                except CanNotFindLegalMoveError:
                    # We don't care about this error we just count the
                    # number of occurences and print a warning
                    num_failed_attempts += 1
                except Exception as exc:
                    print("Totally unexpected exception {} {} on rank {}"
                          "".format(type(exc).__name__, str(exc), self.rank))
                    print("Exceptions here are not handles, due to "
                          "communication overhead")

            print("MC calculation finished on rank: {}".format(self.rank))

            nproc = 1
            # Collect the legal move error from all processes
            if self.mpicomm is not None:
                num_failed_attempts = \
                    self.mpicomm.allreduce(num_failed_attempts)
                nproc = self.mpicomm.Get_size()

            if num_failed_attempts > 0:
                frac_failed = float(num_failed_attempts)/(nsteps * nproc)
                self.log("Number of failed trial moves: {} ({:.1f}%)"
                         "".format(num_failed_attempts, 100 * frac_failed))

            self.log_window_statistics(self.current_window)

            self.log(
                "Acceptance rate in window {}: {}".format(
                    self.current_window, float(
                        self.mc.num_accepted) / self.mc.current_step))

            self.log("Final chemical formula: {}".format(
                self.mc.atoms.get_chemical_formula()))
            self.mc.reset()

            self.save_current_window()
        self.log("Convered all bins: {}".format(self.converged_all_bins))
        self.save()

    @staticmethod
    def dset_name(window):
        """Return a dataset name.

        :param window: Int. Number of the window

        :return: String used as key for the dataset in the HDF5 file
        :rtype: str
        """
        return "window{}".format(window)

    def _collect_results(self, window):
        """
        Collects the results from all processors

        :param int window: Index of the window
        """
        if self.mpicomm is None:
            return
        recv_buf = np.zeros_like(self.data[window])
        self.mpicomm.Allreduce(self.data[window], recv_buf)
        self.data[window][:] = recv_buf

    def _update_data_entry(self, grp, data):
        """Update one data entry.

        :param H5py.Group grp: HDF5 group that should be altered
        :param numpy.ndarray data: New data points
        """
        for key, value in data.items():
            try:
                old_data = np.array(grp[key])
                new_value = old_data + value
                entry = grp[key]
                entry[...] = new_value
            except KeyError:
                # The field does not exist
                grp.create_dataset(key, data=value)

    def get_free_energy(self):
        """Read from file and create free energy curve.

        :return: Merged histogram
        :rtype: list of floats
        """
        # Load content
        data = []
        merged = []
        if self.rank == 0:
            with h5.File(self.fname, "r") as infile:
                for i in range(0, len(self.data)):
                    name = "window{}".format(i)
                    full_name = name + "/hist"
                    data.append(np.array(infile[full_name]))

            merged = self._get_merged_records(data)
        if self.mpicomm is not None:
            data = self.mpicomm.bcast(data, root=0)
            merged = self.mpicomm.bcast(merged, root=0)
        return merged

    def save(self):
        """Save data to HDF5 file."""
        res = self.get_free_energy()

        if self.rank == 0:
            if os.path.exists(self.fname):
                flag = "r+"
            else:
                flag = "w"
            with h5.File(self.fname, flag) as hfile:
                for key, value in res.items():
                    try:
                        data = hfile[key]
                        data[...] = value
                    except KeyError:
                        hfile.create_dataset(key, data=value)

                for bias in self.mc.bias_potentials:
                    if hasattr(bias, "get"):
                        values = bias.get(res["x"])
                        key = str(bias.__class__.__name__)
                        try:
                            data = hfile[key]
                            data[...] = values
                        except KeyError:
                            hfile.create_dataset(key, data=values)
