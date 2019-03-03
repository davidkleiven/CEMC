from cemc.mcmc import ReactionCrdRangeConstraint
from cemc.mcmc import BiasPotential
import numpy as np
import h5py as h5
import time
import os
from ase.db import connect
import traceback


class AdaptiveBiasPotential(BiasPotential):
    """Bias potential intended to be used by AdaptiveBiasReactionPathSampler

    :param list lim: List of length 2 with upper and lower bound of the
        reaction coordinate. Ex. [0.0, 1.0]
    :param int n_bins: Number of bins used in the histogram
    :param float mod_factor: Modification factor. Each time
        the samplers is a bin, this factor is added to the
        potential.
    :param ReactionCrdInitializer reac_init: Initializer to
        which can obtain the reaction coordinate of
        an arbitrary structure
    :param float T: Temperature in Kelvin
    :param Montecarlo mc: Monte Carlo object which samples
        the configurational space
    :param str db_bin_data: Database where one structure in each
        bin will be stored
    """
    def __init__(self, lim=[0.0, 1.0], n_bins=100, mod_factor=0.01,
                 observer=None, T=400, mc=None, db_bin_data="adaptive_bias.db",
                 mpicomm=None, value_name=""):
        from ase.units import kB
        self.xmin = lim[0]
        self.xmax = lim[1]
        self.nbins = n_bins
        self.bias_array = np.zeros(self.nbins)
        self.mod_factor = mod_factor
        self.observer = observer
        self.value_name = value_name
        self.beta = 1.0/(kB*T)
        self.dx = (self.xmax - self.xmin)/self.nbins
        self.mc = mc
        self.db_bin_data = db_bin_data
        self.know_structure_in_bin = np.zeros(self.nbins, dtype=np.uint8)
        self.lowest_active_indx = 0
        self.mpicomm = mpicomm

    @property
    def rank(self):
        if self.mpicomm is None:
            return 0
        return self.mpicomm.Get_rank()

    @property
    def num_proc(self):
        if self.mpicomm is None:
            return 1
        return self.mpicomm.Get_size()

    def average_from_all_proc(self):
        """Average the bias potential from all processors."""
        if self.mpicomm is None:
            return
        recv_buf = np.zeros_like(self.bias_array)
        self.mpicomm.Allreduce(self.bias_array, recv_buf)
        self.bias_array = recv_buf/self.num_proc

    def get_bin(self, value):
        """Return the bin corresponding to value.

        :param float value: Reaction coordinate
        :return: Corresponding bin
        :rtype: int
        """
        return int((value - self.xmin)*self.nbins/(self.xmax - self.xmin))

    def get_value(self, bin_indx):
        """Return the value corresponding to bin."""
        return self.dx*bin_indx + self.xmin

    def update(self):
        """Update the bias potential."""
        x = self.observer.get_current_value()[self.value_name]
        bin_indx = self.get_bin(x)
        if bin_indx < 0 or bin_indx >= self.nbins:
            return
        cur_val = self.get_bias_potential(x)
        self.bias_array[bin_indx] += self.mod_factor
        new_val = self.get_bias_potential(x)

        # We need to make MC aware of that the
        # energy of the current state changes
        # despite the fact that no move is performed
        self.mc.current_energy += (new_val - cur_val)

        # Store one structure in each bin in a database
        if not self.know_structure_in_bin[bin_indx]:
            db = connect(self.db_bin_data)
            db.write(self.mc.atoms, bin_indx=bin_indx, reac_crd=x)
            self.know_structure_in_bin[bin_indx] = 1

    def get_random_structure(self, bin_range):
        """Return a structure from the DB."""
        from random import choice
        db = connect(self.db_bin_data)
        candidates = []
        scond = [("bin_indx", ">", bin_range[0]),
                 ("bin_indx", "<", bin_range[1])]
        for row in db.select(scond):
            candidates.append(row.toatoms())

        if not candidates:
            return None
        return choice(candidates)

    def get_bias_potential(self, value):
        """Return the value of the bias potential.

        :param float value: Reaction coordinate
        :return: Value of the bias potential
        :rtype: float
        """
        bin_indx = self.get_bin(value)
        if bin_indx == self.nbins - 1:
            # Linear interpolation
            betaG2 = self.bias_array[bin_indx]
            betaG1 = self.bias_array[bin_indx-1]
            x1 = self.xmin + (bin_indx - 1)*self.dx
            betaG = (betaG2 - betaG1)*(value - x1)/self.dx + betaG1
        elif bin_indx == self.lowest_active_indx:
            # Linear interpolation
            betaG2 = self.bias_array[self.lowest_active_indx+1]
            betaG1 = self.bias_array[self.lowest_active_indx]
            x1 = self.xmin + bin_indx*self.dx
            betaG = (betaG2 - betaG1)*(value - x1)/self.dx + betaG1
        else:
            # Perform quadratic interpolation
            x0 = self.xmin + bin_indx*self.dx
            x_pluss = x0 + self.dx
            x_minus = x0 - self.dx
            x = np.array([x_minus, x0, x_pluss])
            X = np.zeros((3, 3))
            X[:, 0] = 1.0
            X[:, 1] = x
            X[:, 2] = x**2
            y = np.array([self.bias_array[bin_indx-1],
                          self.bias_array[bin_indx],
                          self.bias_array[bin_indx+1]])
            coeff = np.linalg.solve(X, y)
            betaG = coeff[0] + coeff[1]*value + coeff[2]*value**2
        return betaG/self.beta

    def __call__(self, system_changes):
        """Return the bias potential after changes are applied.

        :param system_changes: Changes to be applied.
            Example: [(10, "Al", "Mg"), (20, "Mg", "Al)]
        :type system_changes: List of tuples

        :return: Bias potential after changes have been applied
        :rtype: float
        """
        # We require initializers that can
        # get apply the system changes
        value = self.observer(system_changes, peak=True)[self.value_name]
        return self.get_bias_potential(value)

    def calculate_from_scratch(self, atoms):
        """Calculate the potential from scratch."""
        cur_val = self.observer.calculate_from_scratch(self.mc.atoms)
        value = cur_val[self.value_name]
        return self.get_bias_potential(value)

    def shift_upper_part_of_bias_array(self, indx, new_value):
        """Shift the upper part of the bias array."""
        value = self.observer.get_current_value()[self.value_name]
        cur_bias = self.get_bias_potential(value)
        diff = new_value - self.bias_array[indx]
        self.bias_array[indx:] += diff
        new_bias = self.get_bias_potential(value)

        # Update the current energy
        self.mc.current_energy += (new_bias - cur_bias)


class AdaptiveBiasReactionPathSampler(object):
    """Sample the free energy along a path by adaptively tuning a bias potential.

    :param Montecarlo mc_obj: Monte Carlo sampler
    :param ReactionCrdInitializer reac_crd_init: Initializer that
        can both set and get the reaction coordinate of an arbitrary
        configuration. Benefitial if this support fast evaluation
        of the reaction coordinate if the proposed system changes
        are supplied.
    :param int n_bins: Number of bins
    :param str data_file: HDF5 file for data backup
    :param list reac_crd: List of length 2 with the upper
        and the lower value of the reaction coordinate.
        If it is possible for the MC sampler to leave the
        region of interest, the MC sampler should have a
        constraint prohibiting this attached.
    :param float mod_factor: Modification factor used to
        update the bias potential
    :param float convergence_factor: If the bin that has been
        is least frequently visited have been visited more than
        convergence_factr*<average visits> the algorithm
        will consider the visit histogram as flat.
    :param int save_interval: Interval between writing backup to file
        in seconds
    :param int log_msg_interval: Interval in seconds between every
        time a status message is printed.
    :param str db_struct: One structure in each bin will be stored
        in a database.
    :param bool delete_db_if_exists: If True any existing DB containing
        structures will be delted prior to the run.
    :param bool ignore_equil_steps: If True MC trial steps that leads to no
        change in the reaction coordinate will not be counted
    """
    def __init__(self, mc_obj=None, observer=None, n_bins=100,
                 data_file="adaptive_bias_path_sampler.h5",
                 react_crd=[0.0, 1.0], mod_factor=0.01, convergence_factor=0.8,
                 save_interval=600, log_msg_interval=30,
                 db_struct="adaptive_bias.db", delete_db_if_exists=False,
                 mpicomm=None, check_convergence_interval=10000,
                 check_user_input=True, ignore_equil_steps=True,
                 react_crd_name=""):

        self.bias = AdaptiveBiasPotential(lim=react_crd, n_bins=n_bins,
                                          mod_factor=mod_factor,
                                          observer=observer, T=mc_obj.T,
                                          mc=mc_obj, db_bin_data=db_struct,
                                          value_name=react_crd_name)
        self.ignore_equil_steps = ignore_equil_steps
        self.mc = mc_obj
        self.mc.attach(observer)
        self.move_accpted = False
        self.current_reac_value = observer.get_current_value()[react_crd_name]
        self.mpicomm = mpicomm
        self.visit_histogram = np.zeros(n_bins, dtype=int)
        self.convergence_factor = convergence_factor
        self.save_interval = save_interval
        self.last_save = time.time()
        self.output_every = log_msg_interval
        self.current_step = 0.0
        self.current_min_val = 0.0
        self.current_max_val = 0.0
        self.average_visits = 0.0
        self.current_mc_step = 0
        self.last_visited_bin = 0
        self.data_file = data_file
        self.load_bias()
        self.mc.add_bias(self.bias)

        # Variables related to adaptive windows
        self.rng_constraint = None
        for cnst in self.mc.constraints:
            if isinstance(cnst, ReactionCrdRangeConstraint):
                self.rng_constraint = cnst

        self.min_window_width = 10
        self.connection = None
        self.current_min_bin = 0
        self.check_convergence_interval = check_convergence_interval
        self.mpicomm = mpicomm

        # Make sure that each processor has a different seed
        from cemc.mcmc.mpi_tools import set_seeds
        set_seeds(self.mpicomm)

        if check_user_input:
            self.give_input_advise()
        if delete_db_if_exists and os.path.exists(db_struct) and self.is_master:
            os.remove(db_struct)

    def give_input_advise(self):
        """Check the input such to help users select good parameters."""
        warning_printed = False
        if self.mpicomm is not None and self.check_convergence_interval < 10000:
            warning_printed = True
            self.log("Warning! Check convergence is set to {}."
                     "This process involves collective communication. "
                     "Therefore it is recommended to increase this number "
                     "at least beyound 10000"
                     "".format(self.check_convergence_interval))

        if self.output_every < 10:
            warning_printed = True
            self.log("Warning! Do you really want to log with as little as "
                     "{} sec interval?".format(self.output_every))

        if self.save_interval < 60:
            warning_printed = True
            self.log("Warning! Do you really want to save backup every "
                     "{} sec? It is recommended to increase this."
                     "".format(self.save_interval))

        if warning_printed:
            self.log("To supress these messages set check_user_input=False")

    @property
    def support_adaptive_windows(self):
        return isinstance(self.rng_constraint, ReactionCrdRangeConstraint)

    def parameter_summary(self):
        """Print a summary of the current parameters."""
        self.log("Temperature: {}".format(self.mc.T))
        self.log("Modification factor: {}".format(self.bias.mod_factor))
        self.log("Reaction coordinate: [{}, {})".format(self.bias.xmin,
                                                        self.bias.xmax))
        self.log("Save every: {} min".format(self.save_interval/60))
        self.log("Log message every: {} sec".format(self.output_every))
        self.log("Support adaptive windows: {}"
                 "".format(self.support_adaptive_windows))

    def load_bias(self):
        """Try to load the bias potential from file."""
        if not os.path.exists(self.data_file):
            return

        with h5.File(self.data_file, 'r') as hfile:
            if "bias" in hfile.keys():
                data = np.array(hfile["bias"])
                self.bias.bias_array = data
        self.log("Bias loaded from {}".format(self.data_file))

        # Subtract of the an overall constant
        self.bias.bias_array -= self.bias.bias_array[0]

    def update(self):
        """Update the history."""
        trial_alter = self._trial_move_alter_reac_crd(self.mc.trial_move)
        if self.ignore_equil_steps and not trial_alter:
            return
        self.bias.update()
        value = self.bias.observer.get_current_value()[self.bias.value_name]
        bin_indx = self.bias.get_bin(value)
        self.last_visited_bin = bin_indx
        self.visit_histogram[bin_indx] += 1
        self.current_reac_value = value

    def _first_non_converged_bin(self):
        """Return the first bin that is not converged."""
        start = self.current_min_bin+self.min_window_width
        for indx in range(start, len(self.visit_histogram)):
            active = self.visit_histogram[self.current_min_bin:indx]
            mean = np.mean(active)
            minval = np.min(active)

            if minval < self.convergence_factor*mean:
                return indx
        return -1

    def _make_energy_curve_continuous(self):
        """Make the energy curve continuous."""
        if self.connection is None:
            return

        assert self.current_min_bin == self.connection["bin"]

        self.bias.shift_upper_part_of_bias_array(self.connection["bin"],
                                                 self.connection["value"])
        self.connection["value"] = self.bias.bias_array[self.connection["bin"]]

    def update_active_window(self):
        """If support adaptive window this will update the relevant part."""
        mean = np.mean(self.visit_histogram[self.current_min_bin:])
        if mean < 1:
            return False
        minval = np.min(self.visit_histogram[self.current_min_bin:])
        self.current_max_val = np.max(self.visit_histogram)
        self.current_min_val = minval
        self.average_visits = mean

        indx = self._first_non_converged_bin()
        if indx == -1:
            self._make_energy_curve_continuous()
            self.connection = None
            return True

        local_conv_ok = abs(indx - self.current_min_bin) > self.min_window_width
        remaining_ok = (indx < len(self.visit_histogram) - self.min_window_width - 1)
        if local_conv_ok and remaining_ok:
            new_structure = self.bias.get_random_structure([indx-1, len(self.visit_histogram)])
            if new_structure is None:
                # We don't now any valid structure so we can't do anything
                return False

            self._make_energy_curve_continuous()
            self.current_min_bin = indx - 1
            self.bias.lowest_active_indx = self.current_min_bin
            self.connection = {"bin": indx - 1,
                               "value": self.bias.bias_array[indx-1]}
            current_range = self.rng_constraint.range
            current_range[0] = self.bias.get_value(self.current_min_bin)
            self.rng_constraint.update_range(current_range)

            # Update the symbols
            symbs = [atom.symbol for atom in new_structure]
            self.mc.set_symbols(symbs)

            # Enforce a calculation of the reaction coordinate
            cur_val = self.bias.observer.calculate_from_scratch(self.mc.atoms)
            value = cur_val[self.bias.value_name]
            self.current_reac_value = value
            self.log("Window shrinked")
            self.log("New value: {}. New range: [{}, {})"
                     "".format(value, current_range[0], current_range[1]))
            self.log("Initialized to bin: {}".format(self.bias.get_bin(value)))
            if value < current_range[0] or value >= current_range[1]:
                raise RuntimeError("System outside window after update!")
        return False

    def converged(self):
        self.synchronize()
        if self.support_adaptive_windows:
            return self.update_active_window()
        else:
            mean = np.mean(self.visit_histogram)
            minval = np.min(self.visit_histogram)
            self.current_max_val = np.max(self.visit_histogram)
            self.current_min_val = minval
            self.average_visits = mean
            return minval > self.convergence_factor*mean and minval > 1

    def log(self, msg):
        """Log a message.

        :param str msg: Message to be logged
        """
        if self.is_master:
            print(msg)

    def progress_message(self):
        """Output a progress message."""
        num_visited = np.count_nonzero(self.visit_histogram > 0)
        acc_rate = int(100*self.mc.num_accepted/self.mc.current_step)
        self.log("Num MC steps: {}".format(self.current_mc_step))
        self.log("Visits: Min: {}. Max: {}. Avg: {}. Num visited: {}. "
                 "Last visited bin: {}. Acc. ratio: {}%. Lower cut: {}"
                 "".format(self.current_min_val, self.current_max_val,
                           self.average_visits, num_visited,
                           self.last_visited_bin, acc_rate,
                           self.current_min_bin))
        self.log("Formula: {}".format(self.mc.atoms.get_chemical_formula()))

    @property
    def rank(self):
        if self.mpicomm is None:
            return 0
        return self.mpicomm.Get_rank()

    @property
    def num_proc(self):
        if self.mpicomm is None:
            return 1
        return self.mpicomm.Get_size()

    @property
    def is_master(self):
        return self.rank == 0

    def barrier(self):
        if self.mpicomm is None:
            return
        self.mpicomm.barrier()

    def synchronize(self):
        """Syncronize the data array from all processors."""
        if self.mpicomm is None:
            return

        self.bias.average_from_all_proc()
        recv_buf = np.zeros_like(self.visit_histogram)
        self.mpicomm.Allreduce(self.visit_histogram, recv_buf)
        self.visit_histogram = recv_buf/self.num_proc

    def save(self):
        """Save records to the HDF5 file."""
        self.synchronize()
        if self.is_master:
            if os.path.exists(self.data_file):
                flag = "r+"
            else:
                flag = "w"

            # We want to always write the continuous potential to file
            bias_array = self.bias.bias_array.copy()
            if self.connection is not None:
                diff = bias_array[self.connection["bin"]] - \
                        self.connection["value"]

                bias_array[self.connection["bin"]:] -= diff

            with h5.File(self.data_file, flag) as hfile:
                if "visits" in hfile.keys():
                    data = hfile["visits"]
                    data[...] = self.visit_histogram
                else:
                    hfile.create_dataset("visits", data=self.visit_histogram)

                if "bias" in hfile.keys():
                    data = hfile["bias"]
                    data[...] = bias_array
                else:
                    hfile.create_dataset("bias", data=bias_array)

                if "x" not in hfile.keys():
                    x = np.linspace(self.bias.xmin, self.bias.xmax,
                                    len(self.visit_histogram))
                    hfile.create_dataset("x", data=x)
            self.log("Current state written to {}".format(self.data_file))
        self.barrier()

    def run(self):
        """Run simulation."""
        self.parameter_summary()
        conv = False
        now = time.time()
        try:
            while not conv:
                energy, self.move_accpted = self.mc._mc_step()
                self.current_mc_step += 1
                self.update()
                if time.time() - now > self.output_every:
                    self.progress_message()
                    now = time.time()

                if time.time() - self.last_save > self.save_interval:
                    self.save()
                    self.last_save = time.time()

                # Check convergence only occationally as this involve
                # collective communication
                if self.current_mc_step % self.check_convergence_interval == 0:
                    conv = self.converged()
        except Exception as exc:
            print(traceback.format_exc())
            print("Rank {}: {}".format(self.rank, str(exc)))

    def _trial_move_alter_reac_crd(self, trial_move):
        """Return True if the trial move alter the reaction coordinate."""
        obs = self.bias.observer
        if self.move_accpted:
            trial_val = obs.get_current_value()[self.bias.value_name]
        else:
            trial_val = obs(trial_move, peak=True)[self.bias.value_name]
        tol = 1E-6
        return abs(self.current_reac_value - trial_val) > tol
