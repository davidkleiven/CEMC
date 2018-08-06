# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division

import numpy as np
import ase.units as units
import time
import logging
from mpi4py import MPI
from scipy import stats
from matplotlib import pyplot as plt
from ase.units import kJ, mol
from cemc.mcmc import mpi_tools
from cemc.mcmc.exponential_filter import ExponentialFilter
from cemc.mcmc.averager import Averager


class DidNotReachEquillibriumError(Exception):
    def __init__(self, msg):
        super(DidNotReachEquillibriumError, self).__init__(msg)


class TooFewElementsError(Exception):
    def __init__(self, msg):
        super(TooFewElementsError, self).__init__(msg)


class CanNotFindLegalMoveError(Exception):
    def __init__(self, msg):
        super(CanNotFindLegalMoveError, self).__init__(msg)


class Montecarlo(object):
    """
    Class for running Monte Carlo at fixed composition

    :param atoms: ASE atoms object (with CE calculator attached!)
    :param temp: Temperature of Monte Carlo simulation in Kelvin
    :param indeces: List of atoms involved Monte Carlo swaps. default is all atoms (currently this has no effect!).
    :param mpicomm: MPI communicator object
    :param logfile: Filename for logging (default is logging to console)
    :param plot_debug: If True it will create some diagnositc plots during equilibration
    """

    def __init__(
            self,
            atoms,
            temp,
            indeces=None,
            mpicomm=None,
            logfile="",
            plot_debug=False,
            min_acc_rate=0.0):
        self.name = "MonteCarlo"
        self.atoms = atoms
        self.T = temp
        self.min_acc_rate = min_acc_rate
        if indeces is None:
            self.indeces = range(len(self.atoms))
        else:
            self.indeces = indeces

        self.observers = []  # List of observers that will be called every n-th step
        # similar to the ones used in the optimization routines

        self.constraints = []
        self.max_allowed_constraint_pass_attempts = 10000

        self.current_step = 0
        self.num_accepted = 0
        self.status_every_sec = 30
        self.atoms_indx = {}
        self.symbols = []
        self._build_atoms_list()
        E0 = self.atoms.get_calculator().get_energy()
        self.current_energy = E0
        self.new_energy = self.current_energy
        self.mean_energy = Averager(ref_value=E0)
        self.energy_squared = Averager(ref_value=E0)
        self.mpicomm = mpicomm
        self.rank = 0
        self.energy_bias = 0.0
        self.update_energy_bias = True

        if (self.mpicomm is not None):
            self.rank = self.mpicomm.Get_rank()
        self.logger = logging.getLogger("MonteCarlo")
        self.logger.setLevel(logging.DEBUG)
        if (logfile == ""):
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self.flush_log = ch.flush
        else:
            ch = logging.FileHandler(logfile)
            ch.setLevel(logging.INFO)
            self.flush_log = ch.emit
        if (not self.logger.handlers):
            self.logger.addHandler(ch)

        # Some member variables used to update the atom tracker, only relevant
        # for canonical MC
        self.rand_a = 0
        self.rand_b = 0
        self.selected_a = 0
        self.selected_b = 0
        self.corrtime_energies = []  # Array of energies used to estimate the correlation time
        self.correlation_info = None
        self.plot_debug = plot_debug
        # Set to false if pyplot should not block when plt.show() is called
        self.pyplot_block = True
        self._linear_vib_correction = None
        self.is_first = True
        self.filter = ExponentialFilter(
            min_time=0.2 * len(self.atoms), max_time=20 * len(self.atoms), n_subfilters=10)
        self.accept_first_trial_move_after_reset = True

    @property
    def linear_vib_correction(self):
        return self._linear_vib_correction

    @linear_vib_correction.setter
    def linear_vib_correction(self, linvib):
        self._linear_vib_correction = linvib
        self.atoms._calc.linear_vib_correction = linvib

    def _check_symbols(self):
        """
        Checks that there is at least to different symbols
        """
        symbs = [atom.symbol for atom in self.atoms]
        count = {}
        for symb in symbs:
            if symb not in count:
                count[symb] = 1
            else:
                count[symb] += 1

        # Verify that there is at two elements with more that two symbols
        if len(count.keys()) < 2:
            raise TooFewElementsError(
                "There is only one element in the given atoms object!")
        n_elems_more_than_2 = 0
        for key, value in count.items():
            if value >= 2:
                n_elems_more_than_2 += 1
        if n_elems_more_than_2 < 2:
            raise TooFewElementsError(
                "There is only one element that has more than one atom")

    def log(self, msg, mode="info"):
        """
        Logs the message as info
        """
        allowed_modes = ["info", "warning"]
        if (mode not in allowed_modes):
            raise ValueError("Mode has to be one of {}".format(allowed_modes))

        if (self.rank != 0):
            return
        if (mode == "info"):
            self.logger.info(msg)
        elif (mode == "warning"):
            self.logger.warning(msg)

    def _no_constraint_violations(self, system_changes):
        """
        Checks if the proposed moves violates any of the constraints

        :param system_changes: Changes of the proposed move
            see :py:class:`cemc.mcmc.mc_observers.MCObserver`
        """
        for constraint in self.constraints:
            if not constraint(system_changes):
                return False
        return True

    def reset(self):
        """
        Reset all member variables to their original values
        """
        for interval, obs in self.observers:
            obs.reset()

        self.filter.reset()
        self.current_step = 0
        self.num_accepted = 0
        self.mean_energy.clear()
        self.energy_squared.clear()
        #self.correlation_info = None
        self.corrtime_energies = []
        if (self.accept_first_trial_move_after_reset):
            self.is_first = True

    def _include_vib(self):
        """
        Includes the vibrational ECIs in the CE ECIs
        """
        self.atoms._calc.include_linvib_in_ecis(self.T)

    def _build_atoms_list(self):
        """
        Creates a dictionary of the indices of each atom which is used to
        make sure that two equal atoms cannot be swapped
        """
        self.atoms_indx = {}
        for atom in self.atoms:
            if (atom.symbol not in self.atoms_indx.keys()):
                self.atoms_indx[atom.symbol] = [atom.index]
            else:
                self.atoms_indx[atom.symbol].append(atom.index)
        self.symbols = list(self.atoms_indx.keys())

    def _update_tracker(self, system_changes):
        """
        Update the atom tracker
        """
        symb_a = system_changes[0][1]
        symb_b = system_changes[1][1]
        self.atoms_indx[symb_a][self.selected_a] = self.rand_b
        self.atoms_indx[symb_b][self.selected_b] = self.rand_a

    def add_constraint(self, constraint):
        """
        Add a new constraint to the sampler

        :param constraint: Instance of :py:class:`cemc.mcmc.mc_constraints.MCConstraint`
        """
        self.constraints.append(constraint)

    def attach(self, obs, interval=1):
        """
        Attach observers that is called on each MC step
        and receives information of which atoms get swapped

        :param obs: Instance of the MCObserver class
        :param interval: the obs.__call__ method is called at mc steps separated by interval
        """
        if (callable(obs)):
            self.observers.append((interval, obs))
        else:
            raise ValueError("The observer has to be a callable class!")

    def _get_var_average_energy(self):
        """
        Returns the variance of the average energy, taking into account
        the auto correlation time
        """

        # First collect the energies from all processors
        self._collect_energy()
        U = self.mean_energy.mean
        E_sq = self.energy_squared.mean
        var = (E_sq - U**2)
        nproc = 1
        if (self.mpicomm is not None):
            nproc = self.mpicomm.Get_size()

        if (var < 0.0):
            self.log(
                "Variance of energy is smaller than zero. (Probably due to numerical precission)",
                mode="warning")
            self.log("Variance of energy : {}".format(var))
            var = np.abs(var)

        if (
                self.correlation_info is None or not self.correlation_info["correlation_time_found"]):
            return var / (self.current_step * nproc)

        tau = self.correlation_info["correlation_time"]
        if (tau < 1.0):
            tau = 1.0
        return 2.0 * var * tau / (self.current_step * nproc)

    def current_energy_without_vib(self):
        """
        Returns the current energy without the contribution from vibrations
        """
        return self.current_energy - \
            self.atoms._calc.vib_energy(self.T) * len(self.atoms)

    def _estimate_correlation_time(self, window_length=1000, restart=False):
        """
        Estimates the correlation time
        """
        self.log("*********** Estimating correlation time ***************")
        if (restart):
            self.corrtime_energies = []
        for i in range(window_length):
            self._mc_step()
            self.corrtime_energies.append(self.current_energy_without_vib())

        mean = np.mean(self.corrtime_energies)
        energy_dev = np.array(self.corrtime_energies) - mean
        var = np.var(energy_dev)
        auto_corr = np.correlate(energy_dev, energy_dev, mode="full")
        auto_corr = auto_corr[int(len(auto_corr) / 2):]

        # Find the point where the ratio between var and auto_corr is 1/2
        self.correlation_info = {
            "correlation_time_found": False,
            "correlation_time": 0.0,
            "msg": ""
        }

        if (var == 0.0):
            self.correlation_info["msg"] = "Zero variance leads to infinite correlation time"
            self.log(self.correlation_info["msg"])
            self.correlation_info["correlation_time_found"] = True
            self.correlation_info["correlation_time"] = window_length
            return self.correlation_info

        auto_corr /= (window_length * var)
        if (np.min(auto_corr) > 0.5):
            self.correlation_info["msg"] = "Window is too short. Add more samples"
            self.log(self.correlation_info["msg"])
            self.correlation_info["correlation_time"] = window_length
            return self.correlation_info

        # See:
        # Van de Walle, A. & Asta, M.
        # Self-driven lattice-model Monte Carlo simulations of alloy thermodynamic properties and
        # phase diagrams Modelling and Simulation
        # in Materials Science and Engineering, IOP Publishing, 2002, 10, 521
        # for details on  the notation
        indx = 0
        for i in range(len(auto_corr)):
            if (auto_corr[i] < 0.5):
                indx = i
                break
        rho = 2.0**(-1.0 / indx)
        tau = -1.0 / np.log(rho)
        self.correlation_info["correlation_time"] = tau
        self.correlation_info["correlation_time_found"] = True
        self.log("Estimated correlation time: {}".format(tau))

        if (self.plot_debug):
            gr_spec = {"hspace": 0.0}
            fig, ax = plt.subplots(nrows=2, gridspec_kw=gr_spec, sharex=True)
            x = np.arange(len(self.corrtime_energies))
            ax[0].plot(x, np.array(self.corrtime_energies) * mol / kJ)
            ax[0].set_ylabel("Energy (kJ/mol)")
            ax[1].plot(x, auto_corr, lw=3)
            ax[1].plot(x, np.exp(-x / tau))
            ax[1].set_xlabel("Number of MC steps")
            ax[1].set_ylabel("ACF")
            plt.show(block=self.pyplot_block)
        return self.correlation_info

    def _composition_reached_equillibrium(
            self,
            prev_composition,
            var_prev,
            confidence_level=0.05):
        """
        Returns True if the composition reached equillibrium.
        Default the simulation runs at fixed composition so
        this function just returns True

        :param prev_composition: Previous composition
        :param var_prev: Variance of the composition
        :param confidence_level: Confidence level used for testing
        """
        return True, prev_composition, var_prev, 0.0

    def _equillibriate(
            self,
            window_length="auto",
            confidence_level=0.05,
            maxiter=1000,
            mode="stat_equiv"):
        """
        Runs the MC until equillibrium is reached

        :param window_length: the length of the window used to compare averages
                     if window_lenth='auto' then the length of window is set to
                     10*len(self.atoms)
        :param confidence_level: Confidence level used in hypothesis testing
                     The question asked in the hypothesis testing is:
                     Given that the two windows have the same average
                     and the variance observed (null hypothesis is correct),
                     what is the probability of observering an even larger
                     difference between the average values in the to windows?
                     If the probability of observing an even larger difference
                     is considerable, then we conclude that the system
                     has reaced equillibrium.
                     confidence_level=0.05 means that the algorithm will
                     terminate if the probability of observering an even
                     larger difference is larger than 5 percent.

                     NOTE: Since a Markiv Chain has large correlation
                     the variance is underestimated. Hence, one can
                     safely use a lower confidence level.

        :param maxiter: The maximum number of windows it will try to sample.
            If it reaches this number of iteration the algorithm will
            raise an error
        """
        allowed_modes = ["stat_equiv", "fixed"]
        if mode not in allowed_modes:
            raise ValueError(
                "Equilibration mode has to be one of {}".format(allowed_modes))

        if (window_length == "auto"):
            window_length = 10 * len(self.atoms)
        nproc = 1
        if (self.mpicomm is not None):
            nproc = self.mpicomm.Get_size()

        self.reset()
        if mode == "fixed":
            self.log("Equilibriating with {} MC steps".format(window_length))
            for _ in range(window_length):
                self._mc_step()
            return

        E_prev = None
        var_E_prev = None
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0 - confidence_level)
        number_of_iterations = 1
        self.log("Equillibriating system")
        self.log("Confidence level: {}".format(confidence_level))
        self.log("Percentiles: {}, {}".format(min_percentile, max_percentile))

        if self.name == "SGCMonteCarlo":
            self.log(
                "{:10} {:10} {:10} {:10} {:10} {:10}".format(
                    "Energy",
                    "std.dev",
                    "delta E",
                    "quantile",
                    "Singlets",
                    "Quantile (compositions)"))
        else:
            self.log(
                "{:10} {:10} {:10} {:10}".format(
                    "Energy",
                    "std.dev",
                    "delta E",
                    "quantile"))

        all_energies = []
        means = []
        composition = []
        var_comp = []
        energy_conv = False
        for i in range(maxiter):
            number_of_iterations += 1
            self.reset()
            for i in range(window_length):
                self._mc_step()
                self.mean_energy += self.current_energy_without_vib()
                self.energy_squared += self.current_energy_without_vib()**2
                if (self.plot_debug):
                    all_energies.append(
                        self.current_energy_without_vib() / len(self.atoms))
            self._collect_energy()
            E_new = self.mean_energy.mean
            means.append(E_new)
            var_E_new = self._get_var_average_energy()
            comp_conv, composition, var_comp, comp_quant = self._composition_reached_equillibrium(
                composition, var_comp, confidence_level=confidence_level)
            if (E_prev is None):
                E_prev = E_new
                var_E_prev = var_E_new
                continue

            var_diff = var_E_new + var_E_prev
            diff = E_new - E_prev
            if (var_diff < 1E-6):
                self.log("Zero variance. System does not move.")
                z_diff = 0.0
                comp_conv = True
            else:
                z_diff = diff / np.sqrt(var_diff)

            if (len(composition) == 0):
                self.log(
                    "{:10.2f} {:10.6f} {:10.6f} {:10.2f}".format(
                        E_new, var_E_new, diff, z_diff))
            else:
                self.log("{:10.2f} {:10.6f} {:10.6f} {:10.2f} {} {:10.2f}".format(
                    E_new, var_E_new, diff, z_diff, composition, comp_quant))
            # self.logger.handlers[0].flush()
            # self.flush_log()
            #print ("{:10.2f} {:10.6f} {:10.6f} {:10.2f}".format(E_new,var_E_new,diff, z_diff))
            if((z_diff < max_percentile) and (z_diff > min_percentile)):
                energy_conv = True

            # Allreduce use the max value which should yield True if any is
            # True
            if (self.mpicomm is not None):
                eng_conv = np.array(energy_conv, dtype=np.uint8)
                eng_conv_recv = np.zeros(1, dtype=np.uint8)
                self.mpicomm.Allreduce(eng_conv, eng_conv_recv, op=MPI.MAX)
                energy_conv = eng_conv_recv[0]

                comp_conv_arr = np.array(comp_conv, dtype=np.uint8)
                comp_conv_recv = np.zeros(1, dtype=np.uint8)
                self.mpicomm.Allreduce(
                    comp_conv_arr, comp_conv_recv, op=MPI.MAX)
                comp_conv = comp_conv_recv[0]

            if (energy_conv and comp_conv):
                self.log(
                    "System reached equillibrium in {} mc steps".format(
                        number_of_iterations * window_length))
                self.mean_energy.clear()
                self.energy_squared.clear()
                self.current_step = 0

                if (len(composition) > 0):
                    self.log(
                        "Singlet values at equillibrium: {}".format(composition))

                if (self.plot_debug):
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(np.array(all_energies) * mol / kJ)
                    start = 0
                    for i in range(len(means)):
                        ax.plot([start, start + window_length],
                                [means[i], means[i]], color="#fc8d62")
                        ax.plot(start, means[i], "o", color="#fc8d62")
                        ax.axvline(
                            x=start + window_length,
                            color="#a6d854",
                            ls="--")
                        start += window_length
                    ax.set_xlabel("Number of MC steps")
                    ax.set_ylabel("Energy (kJ/mol)")
                    plt.show(block=self.pyplot_block)
                return

            E_prev = E_new
            var_E_prev = var_E_new

        raise DidNotReachEquillibriumError(
            "Did not manage to reach equillibrium!")

    def _has_converged_prec_mode(
            self,
            prec=0.01,
            confidence_level=0.05,
            log_status=False):
        """
        Returns True if the simulation has converged in the precission mode
        """
        percentile = stats.norm.ppf(1.0 - confidence_level)
        var_E = self._get_var_average_energy()
        converged = (var_E < (prec * len(self.atoms) / percentile)**2)

        if (log_status):
            std_E = np.sqrt(var_E)
            criteria = prec * len(self.atoms) / percentile
            self.log(
                "Current energy std: {}. Convergence criteria: {}".format(
                    std_E, criteria))

        if (self.mpicomm is not None):
            # Make sure that all processors has the same converged flag
            send_buf = np.zeros(1)
            recv_buf = np.zeros(1)
            send_buf[0] = converged
            self.mpicomm.Allreduce(send_buf, recv_buf)
            converged = (recv_buf[0] == self.mpicomm.Get_size())
        return converged

    def _on_converged_log(self):
        """
        Returns message that is printed to the logger after the run
        """
        U = self.mean_energy.mean
        var_E = self._get_var_average_energy()
        self.log("Total number of MC steps: {}".format(self.current_step))
        self.log("Final mean energy: {} +- {}%".format(U,
                                                       np.sqrt(var_E) / np.abs(U)))
        self.log(self.filter.status_msg(
            std_value=np.sqrt(var_E * len(self.atoms))))
        exp_extrapolate = self.filter.exponential_extrapolation()
        self.log("Exponential extrapolation: {}".format(exp_extrapolate))

    def _distribute_correlation_time(self):
        """
        Distrubutes the correlation time to all the processes
        """
        if (self.mpicomm is None):
            return

        # Check if any processor has found an equillibriation time
        corr_time_found = self.mpicomm.gather(
            self.correlation_info["correlation_time_found"], root=0)
        all_corr_times = self.mpicomm.gather(
            self.correlation_info["correlation_time"], root=0)
        self.log("Correlation time on all processors: {}".format(all_corr_times))
        corr_time = self.correlation_info["correlation_time"]
        found = False
        if (self.rank == 0):
            found = np.any(corr_time_found)
            corr_time = np.median(all_corr_times)
            self.log(
                "Using correlation time: {} on all processors".format(corr_time))
        corr_time = self.mpicomm.bcast(corr_time, root=0)
        corrtime_found = self.mpicomm.bcast(found, root=0)
        self.correlation_info["correlation_time_found"] = found
        self.correlation_info["correlation_time"] = corr_time

    def _atleast_one_reached_equillibrium(self, reached_equil):
        """
        Handles the case when a few processors did not reach equillibrium
        The behavior is that if one processor reached equillibrium the
        simulation can continue

        :param reached_equil: Flag True if equillibrium was reached. False otherwise
        """
        if (self.mpicomm is None):
            return reached_equil

        all_equils = self.mpicomm.gather(reached_equil, root=0)
        at_least_one = False
        if (self.rank == 0):
            if (np.any(all_equils)):
                at_least_one = True
        # Send the result to the other processors
        at_least_one = self.mpicomm.bcast(at_least_one, root=0)
        return at_least_one

    def runMC(
            self,
            mode="fixed",
            steps=10,
            verbose=False,
            equil=True,
            equil_params={},
            prec=0.01,
            prec_confidence=0.05):
        """Run Monte Carlo simulation

        :param steps: Number of steps in the MC simulation
        :param verbose: If True information is printed on each step
        :param equil: If the True the MC steps will be performed until equillibrium is reached
        :param equil_params: Dictionary of parameters used in the equillibriation routine
                             See the doc-string of :py:meth:`cemc.mcmc.Montecarlo._equillibriate` for more
                             information
        :param prec: Precission of the run. The simulation terminates when
            <E>/std(E) < prec with a confidence given prec_confidence
        :param prec_confidence: Confidence level used when determining if enough
                          MC samples have been collected
        """
        #print ("Proc start MC: {}".format(self.rank))
        # Check the number of different elements are correct to avoid
        # infinite loops
        self._check_symbols()

        if (self.mpicomm is not None):
            self.mpicomm.barrier()
        allowed_modes = ["fixed", "prec"]
        if (mode not in allowed_modes):
            raise ValueError("Mode has to be one of {}".format(allowed_modes))

        # Include vibrations in the ECIS, does nothing if no vibration ECIs are
        # set
        self._include_vib()

        # Atoms object should have attached calculator
        # Add check that this is show
        self._mc_step()

        mpi_tools.set_seeds(self.mpicomm)
        totalenergies = []
        totalenergies.append(self.current_energy)
        start = time.time()
        prev = 0
        self.current_step = 0

        if (equil):
            reached_equil = True
            res = self._estimate_correlation_time(restart=True)
            if (not res["correlation_time_found"]):
                res["correlation_time"] = 1000
                res["correlation_time_found"] = True
            self._distribute_correlation_time()

            try:
                self._equillibriate(**equil_params)
            except DidNotReachEquillibriumError:
                reached_equil = False

            at_lest_one_proc_reached_equil = self._atleast_one_reached_equillibrium(
                reached_equil)
            # This exception is MPI safe, as the function _atleast_one_reached_equillibrium broadcasts
            # the result to all the other processors
            if (not at_lest_one_proc_reached_equil):
                raise DidNotReachEquillibriumError()

        check_convergence_every = 1
        next_convergence_check = len(self.atoms)
        if (mode == "prec"):
            # Estimate correlation length
            res = self._estimate_correlation_time(restart=True)
            while (not res["correlation_time_found"]):
                res = self._estimate_correlation_time()
            self.reset()
            self._distribute_correlation_time()
            check_convergence_every = 10 * \
                self.correlation_info["correlation_time"]
            next_convergence_check = check_convergence_every

        #print ( "Proc: {} - {}".format(self.rank,check_convergence_every) )
        # self.current_step gets updated in the _mc_step function
        log_status_conv = True
        self.reset()
        while(self.current_step < steps):
            en, accept = self._mc_step(verbose=verbose)
            self.mean_energy += self.current_energy_without_vib()
            self.energy_squared += self.current_energy_without_vib()**2

            if (time.time() - start > self.status_every_sec):
                ms_per_step = 1000.0 * self.status_every_sec / \
                    float(self.current_step - prev)
                accept_rate = self.num_accepted / float(self.current_step)
                log_status_conv = True
                self.log(
                    "%d of %d steps. %.2f ms per step. Acceptance rate: %.2f" %
                    (self.current_step, steps, ms_per_step, accept_rate))
                prev = self.current_step
                start = time.time()
            if (mode == "prec" and self.current_step > next_convergence_check):
                next_convergence_check += check_convergence_every
                # TODO: Is this barrier nessecary, or does it impact performance?
                #print ("Proc check_conv: {}".format(self.rank))
                if (self.mpicomm is not None):
                    self.mpicomm.barrier()
                converged = self._has_converged_prec_mode(
                    prec=prec, confidence_level=prec_confidence, log_status=log_status_conv)
                log_status_conv = False
                if (converged):
                    self._on_converged_log()
                    break

        if self.current_step >= steps:
            self.log(
                "Reached maximum number of steps ({} mc steps)".format(steps))

        if (self.mpicomm is not None):
            self.mpicomm.barrier()
        return totalenergies

    def _collect_energy(self):
        """
        Sums the energy from each processor
        """
        if (self.mpicomm is None):
            return

        size = self.mpicomm.Get_size()
        self.mean_energy = self.mpicomm.allreduce(self.mean_energy, op=MPI.SUM)
        self.energy_squared = self.mpicomm.allreduce(
            self.energy_squared, op=MPI.SUM)

    def get_thermodynamic(self):
        """
        Compute thermodynamic quantities
        """
        self._collect_energy()
        quantities = {}
        quantities["energy"] = self.mean_energy.mean
        mean_sq = self.energy_squared.mean
        quantities["heat_capacity"] = (
            mean_sq - quantities["energy"]**2) / (units.kB * self.T**2)
        quantities["energy_std"] = np.sqrt(self._get_var_average_energy())
        quantities["temperature"] = self.T
        at_count = self.count_atoms()
        for key, value in at_count.items():
            name = "{}_conc".format(key)
            conc = float(value) / len(self.atoms)
            quantities[name] = conc
        return quantities

    def _get_trial_move(self):
        """
        Perform a trial move by swapping two atoms
        """
        self.rand_a = self.indeces[np.random.randint(0, len(self.indeces))]
        self.rand_b = self.indeces[np.random.randint(0, len(self.indeces))]
        symb_a = self.symbols[np.random.randint(0, len(self.symbols))]
        symb_b = symb_a
        while (symb_b == symb_a):
            symb_b = self.symbols[np.random.randint(0, len(self.symbols))]

        Na = len(self.atoms_indx[symb_a])
        Nb = len(self.atoms_indx[symb_b])
        self.selected_a = np.random.randint(0, Na)
        self.selected_b = np.random.randint(0, Nb)
        self.rand_a = self.atoms_indx[symb_a][self.selected_a]
        self.rand_b = self.atoms_indx[symb_b][self.selected_b]

        # TODO: The MC calculator should be able to have constraints on which
        # moves are allowed. CE requires this some elements are only allowed to
        # occupy some sites
        symb_a = self.atoms[self.rand_a].symbol
        symb_b = self.atoms[self.rand_b].symbol
        system_changes = [(self.rand_a, symb_a, symb_b),
                          (self.rand_b, symb_b, symb_a)]
        return system_changes

    def _accept(self, system_changes):
        """
        Returns True if the trial step is accepted
        """
        self.new_energy = self.atoms._calc.calculate(
            self.atoms, ["energy"], system_changes)
        if (self.is_first):
            self.is_first = False
            return True

        if (self.new_energy < self.current_energy):
            return True
        kT = self.T * units.kB
        energy_diff = self.new_energy - self.current_energy
        probability = np.exp(-energy_diff / kT)
        return np.random.rand() <= probability

    def count_atoms(self):
        """
        Count the number of each species
        """
        atom_count = {key: 0 for key in self.symbols}
        for atom in self.atoms:
            atom_count[atom.symbol] += 1
        return atom_count

    def _mc_step(self, verbose=False):
        """
        Make one Monte Carlo step by swithing two atoms
        """
        self.current_step += 1
        number_of_atoms = len(self.atoms)

        system_changes = self._get_trial_move()
        counter = 0
        while not self._no_constraint_violations(system_changes) and \
                counter < self.max_allowed_constraint_pass_attempts:
            system_changes = self._get_trial_move()

        if counter == self.max_allowed_constraint_pass_attempts:
            msg = "Did not manage to produce a trial move that does not "
            msg += "violate any of the constraints"
            raise CanNotFindLegalMoveError(msg)

        move_accepted = self._accept(system_changes)

        if (move_accepted):
            self.current_energy = self.new_energy
            self.num_accepted += 1
        else:
            # Reset the sytem back to original
            for change in system_changes:
                indx = change[0]
                old_symb = change[1]
                assert (self.atoms[indx].symbol == change[2])
                self.atoms[indx].symbol = old_symb

        # TODO: Wrap this functionality into a cleaning object
        if (hasattr(self.atoms._calc, "clear_history")
                and hasattr(self.atoms._calc, "undo_changes")):
            # The calculator is a CE calculator which support clear_history and
            # undo_changes
            pass
        if (move_accepted):
            self.atoms._calc.clear_history()
        else:
            self.atoms._calc.undo_changes()

        if (move_accepted):
            # Update the atom_indices
            self._update_tracker(system_changes)
        else:
            new_symb_changes = []
            for change in system_changes:
                new_symb_changes.append((change[0], change[1], change[1]))
            system_changes = new_symb_changes
            # system_changes =
            # [(self.rand_a,symb_a,symb_a),(self.rand_b,symb_b,symb_b)] # No
            # changes to the system

        # Execute all observers
        for entry in self.observers:
            interval = entry[0]
            if (self.current_step % interval == 0):
                obs = entry[1]
                obs(system_changes)
        self.filter.add(self.current_energy)
        return self.current_energy, move_accepted
