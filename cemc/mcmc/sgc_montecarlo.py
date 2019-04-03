from cemc.mcmc import montecarlo as mc
from cemc.mcmc.mc_observers import SGCObserver
import numpy as np
from ase.units import kB
from scipy import stats
from cemc.mcmc import mpi_tools

class SGCMonteCarlo(mc.Montecarlo):
    """
    Class for running Monte Carlo in the Semi-Grand Canonical Ensebmle
    (i.e. fixed number of atoms, but varying composition)

    See docstring of :py:class:`cemc.mcmc.Montecarlo`

    :param Atoms atoms: Atoms object (with CE calculator attached!)
    :param float temp: Temperature in kelvin
    :param indeces: Not used
    :type indeces: list of ints or None
    :param list symbols: List of possible symbols for insertion moves
    :param Intracomm mpicomm: MPI communicator object
    :param str logfile: File for logging (default is console window)
    :param bool plot_debug: Generate debugging plots. Recommended to leave this
        as False
    :param float min_acc_rate: If the acceptance rate drops below this value
        the calculation terminates
    """

    def __init__(self, atoms, temp, indeces=None, symbols=None, mpicomm=None,
                 logfile="", plot_debug=False, min_acc_rate=0.0,
                 recycle_waste=False):
        mc.Montecarlo.__init__(self, atoms, temp, indeces=indeces,
                              mpicomm=mpicomm, logfile=logfile,
                              plot_debug=plot_debug, min_acc_rate=min_acc_rate,
                              recycle_waste=recycle_waste)
        if not symbols is None:
            # Override the symbols function in the main class
            self.symbols = symbols

        if len(self.symbols) <= 1:
            raise ValueError("At least 2 symbols have to be specified")
        self.averager = SGCObserver(self.atoms.get_calculator(), self, len(self.symbols)-1)
        self.chem_pots = []
        self.chem_pot_names = []
        self.has_attached_avg = False
        self.name = "SGCMonteCarlo"
        self._chemical_potential = None
        self.chem_pot_in_ecis = False
        self.composition_correlation_time = np.zeros(len(self.symbols)-1)
        self.current_singlets = None

        has_attached_obs = False
        for obs in self.observers:
            if obs.name == "SGCObserver":
                has_attached_obs = True
                self.averager = obs
                break
        if not has_attached_obs:
            self.attach(self.averager)

    def _get_trial_move(self):
        """
        Generate a trial move by flipping the symbol of one atom

        :return: Proposed move
        :rtype: List of tuples
        """
        self.current_singlets = self.atoms.get_calculator().get_singlets()
        indx = np.random.randint(low=0, high=len(self.atoms))
        old_symb = self.atoms[indx].symbol
        new_symb = old_symb
        while new_symb == old_symb:
            new_symb = self.symbols[np.random.randint(low=0,high=len(self.symbols))]
        system_changes = [(indx,old_symb,new_symb)]
        return system_changes

    def _check_symbols(self):
        """
        Override because there are no restriction on the symbols here
        """
        pass

    def _update_tracker(self, system_changes):
        """
        Override the update of the atom tracker.

        The atom tracker is irrelevant in the semi grand canonical ensemble

        :param list system_changes: Accepted system changes
        """
        pass

    def _get_var_average_singlets(self):
        """
        Returns the variance for the average singlets.

        The correlation time is taken into account into account
        """
        self._collect_averager_results()
        N = self.averager.counter
        singlets = self.averager.quantities["singlets"]/N
        singlets_sq = self.averager.quantities["singlets_sq"]/N
        var_n = singlets_sq - singlets**2

        if np.min(var_n) < -1E-5:
            msg = "The computed variances is {}".format(var_n)
            msg += "This is significantly smaller than zero and cannot be "
            msg += "attributed to numerical precission!"
            self.log(msg)

        nproc = 1
        if self.mpicomm is not None:
            nproc = self.mpicomm.Get_size()

        no_corr_info = self.correlation_info is None

        if no_corr_info:
            corr_time_found = False
        else:
            corr_time_found = self.correlation_info["correlation_time_found"]
        if no_corr_info or not corr_time_found:
            return var_n/(N*nproc)

        if not np.all(var_n > 0.0):
            self.logger.warning("Some variance where smaller than zero. "
                                "(Probably due to numerical precission)")
            self.log("Variances: {}".format(var_n))
            var_n = np.abs(var_n)
        tau = self.correlation_info["correlation_time"]
        if tau < 1.0:
            tau = 1.0
        return 2.0*var_n*tau/(N*nproc)

    def _has_converged_prec_mode(self, prec=0.01, confidence_level=0.05,
                                 log_status=False):
        """
        Checks that the averages have converged to the desired precission

        :param float prec: Precision level
        :param float confidence_level: Confidence level for hypothesis testing
        :param bool log_status: If True it will print a message showing the
            variances and convergence criteria

        :return: True/False. If True the simulation has converged
        :rtype: bool
        """
        energy_converged = super(SGCMonteCarlo, self)._has_converged_prec_mode(
            prec=prec, confidence_level=confidence_level,
            log_status=False)
        percentile = stats.norm.ppf(1.0-confidence_level)
        var_n = self._get_var_average_singlets()
        if self.mpicomm is not None:
            var_n /= self.mpicomm.Get_size()
        singlet_converged = (np.max(var_n) < (prec/percentile)**2)

        result = singlet_converged
        if self.mpicomm is not None:
            send_buf = np.zeros(1, dtype=np.uint8)
            recv_buf = np.zeros(1, dtype=np.uint8)
            send_buf[0] = result
            self.mpicomm.Allreduce(send_buf, recv_buf)
            result = (recv_buf[0] == self.mpicomm.Get_size())

        if log_status:
            print("Singlet std: {}".format(np.sqrt(var_n)))
        return result

    def _on_converged_log(self):
        """
        Log the convergence message
        """
        super(SGCMonteCarlo, self)._on_converged_log()
        singlets = self.averager.singlets/self.averager.counter
        var_n = self._get_var_average_singlets()
        var_n = np.abs(var_n)  # Just in case some variances should be negative
        self.log("Thermal averaged singlet terms:")
        for i in range(len(singlets)):
            self.log("{}: {} +- {}%".format(
                self.chem_pot_names[i], singlets[i],
                np.sqrt(var_n[i])/np.abs(singlets[i])))

    def _composition_reached_equillibrium(self, prev_composition, var_prev,
                                          confidence_level=0.05):
        """
        Returns True if the composition reached equillibrium

        See :py:meth:`cemc.mcmc.Montecarlo.composition_reached_equillibrium`

        :return: Converged flag, value of averaged singlets, percentile
        :rtype: bool, dict, float
        """
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)
        nproc = 1
        if self.mpicomm is not None:
            nproc = self.mpicomm.Get_size()
        # Collect the result from the other processes
        # and average them into the values on the root node
        self._collect_averager_results()
        N = self.averager.counter
        singlets = self.averager.singlets/N
        var_n = self._get_var_average_singlets()

        if len(prev_composition) != len(singlets):
            # Prev composition is unknown so makes no sense
            # to check
            return False, singlets, var_n, 0.0

        # Just in case variance should be smaller than zero. Should never
        # happen but can happen due to numerical precission
        var_n[var_n < 0.0] = 0.0

        var_n /= nproc
        diff = singlets - prev_composition
        var_diff = var_n + var_prev
        if len(var_diff[var_diff > 0.0]) == 0:
            return True, singlets, var_n, 0.0
        z = np.abs(diff[var_diff > 0.0])/np.sqrt(var_diff[var_diff > 0.0])
        z = np.max(z)
        converged = False
        if z > min_percentile and z < max_percentile:
            converged = True

        if self.mpicomm is not None:
            # Broadcast the result to the other processors
            converged = self.mpicomm.bcast(converged, root=0)
            singlets = self.mpicomm.bcast(singlets, root=0)
            var_n = self.mpicomm.bcast(var_n, root=0)
        return converged, singlets, var_n, z

    def reset(self):
        """
        Reset the simulation object
        """
        super(SGCMonteCarlo, self).reset()
        self.averager.reset()

    @property
    def chemical_potential(self):
        return self._chemical_potential

    @chemical_potential.setter
    def chemical_potential(self, chem_pot):
        self._chemical_potential = chem_pot
        if self.chem_pot_in_ecis:
            self._reset_eci_to_original(self.atoms.get_calculator().eci)
        self._include_chemical_potential_in_ecis(
            chem_pot, self.atoms.get_calculator().eci)

    def _include_chemical_potential_in_ecis(self, chem_potential, eci):
        """
        Including the chemical potentials in the ecis

        :param dict chem_potential: Chemical potentials
        :param dict eci: Original ECIs

        :return: ECIs with chemical potential included
        :rtype: dict
        """
        self.chem_pots = []
        self.chem_pot_names = []
        keys = list(chem_potential.keys())
        keys.sort()
        for key in keys:
            self.chem_pots.append(chem_potential[key])
            self.chem_pot_names.append(key)
            current_eci = eci.get(key, 0.0)
            eci[key] = current_eci - chem_potential[key]
        self.atoms.get_calculator().update_ecis(eci)
        self.chem_pot_in_ecis = True
        self.current_energy = self.atoms.get_calculator().get_energy()
        return eci

    def _reset_eci_to_original(self, eci_with_chem_pot):
        """
        Resets the ecis to their original value

        :parma dict eci_with_chem_pot: ECIs with chemical potential included
        """
        for name, val in zip(self.chem_pot_names, self.chem_pots):
            eci_with_chem_pot[name] += val
        self.atoms.get_calculator().update_ecis(eci_with_chem_pot)
        self.chem_pot_in_ecis = False
        self.current_energy = self.atoms.get_calculator().get_energy()
        return eci_with_chem_pot

    def reset_ecis(self):
        """
        Return the ECIs
        """
        if self.chem_pot_in_ecis:
            self._reset_eci_to_original(self.atoms.get_calculator().eci)

    def _estimate_correlation_time_composition(self, window_length=1000,
                                               restart=False):
        """
        Estimate the corralation time for energy and composition

        :param int window_length: Number of samples to estimate the correlation
            time from
        :param bool restart: If True, sampling is started from scratch,
            otherwise already sampled values are included.
        """
        mc.Montecarlo._estimate_correlation_time(
            self, window_length=window_length, restart=restart)

        singlets = [[] for _ in range(len(self.symbols)-1)]
        for i in range(window_length):
            self.averager.reset()
            self._mc_step()
            singl = self.averager.singlets
            for i in range(len(singl)):
                singlets[i].append(singl[i])

        corr_times = []
        window_length_too_short = False
        for dset in singlets:
            mean = np.mean(dset)
            centered_dset = np.array(dset)-mean
            corr = np.correlate(centered_dset, centered_dset, mode="full")
            corr = corr[int(len(corr)/2):]
            var = np.var(centered_dset)
            corr /= (var*window_length)
            if np.min(corr) > 0.5:
                window_length_too_short = True
                corr_times.append(window_length)
            else:
                indx = 0
                for i in range(len(corr)):
                    if corr[i] < 0.5:
                        indx = i
                        break
                rho = 2.0**(-1.0/indx)
                tau = -1.0/np.log(rho)
                corr_times.append(tau)

        self.composition_correlation_time = np.array(corr_times)
        if self.mpicomm is not None:
            send_buf = np.zeros(1, dtype=np.uint8)
            recv_buf = np.zeros(1, dtype=np.uint8)
            send_buf[0] = window_length_too_short
            self.mpicomm.Allreduce(send_buf, recv_buf)
            window_length_too_short = (recv_buf[0] >= 1)

        if window_length_too_short:
            msg = "The window length is too short to estimate the correlation "
            msg += " time. Using the window length as correlation time."
            self.log(msg)

        # Collect the correlation times from all processes
        if self.mpicomm is not None:
            recv_buf = np.zeros_like(self.composition_correlation_time)
            size = self.mpicomm.Get_size()
            self.mpicomm.Allreduce(self.composition_correlation_time, recv_buf)
            self.composition_correlation_time = recv_buf/size
        self.log("Correlation time for the compositions:")
        self.log("{}".format(self.composition_correlation_time))

    def runMC(self, mode="fixed", steps=10, verbose=False, chem_potential=None,
              equil=True, equil_params={}, prec_confidence=0.05, prec=0.01):
        """
        Run Monte Carlo simulation.
        See :py:meth:`cemc.mcmc.Montecarlo.runMC`

        :param dict chem_potential: Chemical potentials.
            The keys should correspond to one of the singlet terms.
            A typical form of this is
            {"c1_0":-1.0,c1_1_1.0}
        """
        mpi_tools.set_seeds(self.mpicomm)
        self.reset()
        if self.mpicomm is not None:
            self.mpicomm.barrier()

        if chem_potential is None and self.chemical_potential is None:
            ex_chem_pot = {
                "c1_1": -0.1,
                "c1_2": 0.05
            }
            raise ValueError("No chemicalpotentials given. Has to be "
                             "dictionary of the form {}".format(ex_chem_pot))

        if chem_potential is not None:
            self.chemical_potential = chem_potential
        self.reset()

        # Include vibrations in the ECIS, does nothing if no vibration ECIs are
        # present
        self._include_vib()

        if equil:
            reached_equil = True
            res = self._estimate_correlation_time(restart=True)
            if not res["correlation_time_found"]:
                res["correlation_time_found"] = True
                res["correlation_time"] = 1000
            self._distribute_correlation_time()
            try:
                self._equillibriate(**equil_params)
            except mc.DidNotReachEquillibriumError:
                reached_equil = False
            atleast_one_proc = self._atleast_one_reached_equillibrium(
                reached_equil)

            if not atleast_one_proc:
                raise mc.DidNotReachEquillibriumError()

        self.reset()
        mc.Montecarlo.runMC(self, steps=steps, verbose=verbose, equil=False,
                            mode=mode, prec_confidence=prec_confidence,
                            prec=prec)

    def _collect_averager_results(self):
        """
        If MPI is used, this function collects the results from the averager
        """
        if self.mpicomm is None:
            return

        size = self.mpicomm.Get_size()
        all_res = self.mpicomm.gather(self.averager.quantities, root=0)

        # Check that all processors have performed the same number of steps
        # (which they should)
        same_number_of_steps = True
        msg = ""
        if self.rank == 0:
            for i in range(1,len(all_res)):
                if all_res[i]["counter"] != all_res[0]["counter"]:
                    same_number_of_steps = False
                    msg = "Processor {} have performed a different ".format(i)
                    msg += "number steps compared to 0."
                    msg += "Number of stest {}: ".format(i)
                    msg += "{}. Number of steps 0: {}".format(
                        all_res[i]["counter"], all_res[0]["counter"])
                    break
        same_number_of_steps = self.mpicomm.bcast(same_number_of_steps, root=0)

        if not same_number_of_steps:
            raise RuntimeError(msg)

        par_works = True
        if self.rank == 0:
            par_works = self._parallelization_works(all_res)
        par_works = self.mpicomm.bcast(par_works, root=0)
        if not par_works:
            # This can happen either because the seed on all processors are the
            # same or because the results hav already been collected
            return
            msg = "It seems like exactly the same process is running on "
            msg += "multiple processors!"
            raise RuntimeError(msg)

        # Average all the results from the all the processors
        if self.rank == 0:
            self.averager.quantities = all_res[0]
            for i in range(1, len(all_res)):
                for key, value in all_res[i].items():
                    self.averager.quantities[key] += value

            # Normalize by the number of processors
            for key in self.averager.quantities.keys():
                if key != "energy" and key != "energy_sq":
                    self.averager.quantities[key] /= size

        # Broadcast the averaged results
        self.averager.quantities = self.mpicomm.bcast(self.averager.quantities,
                                                      root=0)

    def singlet2composition(self, avg_singlets):
        """Convert singlets to composition."""
        bf = self.atoms.get_calculator().BC.basis_functions
        matrix = np.zeros((len(self.symbols), len(self.symbols)))

        index = {s: i for i, s in enumerate(self.symbols)}
        for i, b in enumerate(bf):
            for s, col in index.items():
                matrix[i, col] = b[s]

        matrix[-1, :] = 1.0
        rhs = np.zeros(len(self.symbols))
        rhs[:-1] = avg_singlets
        rhs[-1] = 1.0
        x = np.linalg.solve(matrix, rhs)

        res = {}
        for s, i in index.items():
            name = s + "_conc"
            res[name] = x[i]
        return res

    def get_thermodynamic(self, reset_ecis=True):
        """
        Compute thermodynamic quantities

        :param bool reset_ecis: If True, the chemical potential will be
            removed from the ECIs

        :return: Thermodynamic quantities
        :rtype: dict
        """
        self._collect_averager_results()
        N = self.averager.counter
        quantities = {}
        singlets = self.averager.singlets/N
        singlets_sq = self.averager.quantities["singlets_sq"]/N

        quantities["sgc_energy"] = self.averager.energy.mean
        quantities["sgc_heat_capacity"] = self.averager.energy_sq.mean - \
            self.averager.energy.mean**2

        quantities["sgc_heat_capacity"] /= (kB*self.T**2)

        quantities["energy"] = self.averager.energy.mean
        natoms = len(self.atoms)
        for i in range(len(self.chem_pots)):
            quantities["energy"] += self.chem_pots[i]*singlets[i]*natoms

        quantities["temperature"] = self.T
        quantities["n_mc_steps"] = self.averager.counter
        # Add singlets and chemical potential to the dictionary
        for i in range(len(singlets)):
            name = "singlet_{}".format(self.chem_pot_names[i])
            quantities[name] = singlets[i]

            name = "var_singlet_{}".format(self.chem_pot_names[i])
            quantities[name] = singlets_sq[i] - singlets[i]**2

            name = "mu_{}".format(self.chem_pot_names[i])
            quantities[name] = self.chem_pots[i]

        quantities.update(self.meta_info)

        try:
            avg_conc = self.singlet2composition(singlets)
            quantities.update(avg_conc)
        except Exception as exc:
            print("Could not find average singlets!")
            print(exc)

        if reset_ecis:
            self._reset_eci_to_original(self.atoms.get_calculator().eci)
        return quantities

    def _parallelization_works(self, all_res):
        """
        Checks that the entries in all_res are different.
        If not it seems like the same process is running on
        all the processors
        """
        if all_res is None:
            return True

        ref_proc = all_res[-1]  # Use the last processor as reference
        for i in range(0, len(all_res)-1):
            for key in ref_proc.keys():
                if key == "counter":
                    continue

                if isinstance(ref_proc[key], np.ndarray):
                    if not np.allclose(ref_proc[key], all_res[i][key]):
                        return True
                else:
                    if ref_proc[key] != all_res[i][key]:
                        return True
        return False
