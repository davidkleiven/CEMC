from ase.units import kB
import numpy as np


class ParallelTempering(object):
    def __init__(self, mc_obj=None, Tmax=1500.0, Tmin=100.0,
                 temp_scheme_file="temp_scheme.csv", mpicomm=None):
        from cemc.mcmc import Montecarlo
        if not isinstance(mc_obj, Montecarlo):
            raise TypeError("mc_obj has to be of type Montecarlo!")
        # Remove the MPI communicator
        mc_obj.mpicomm = None

        self.mpicomm = mpicomm
        mc_obj.T = Tmax
        self.mc_objs = [mc_obj]
        self.natoms = len(mc_obj.atoms)
        self.rank = 0
        self.active_replica = 0

        if self.mpicomm is not None:
            self.rank = self.mpicomm.Get_rank()
        self.temperature_schedule_fname = temp_scheme_file
        self.Tmax = Tmax
        self.Tmin = Tmin
        self._init_temperature_scheme()
        self._adapt_temperature_schedule_to_mpi_size()

    def _log(self, msg):
        if self.rank == 0:
            print(msg)

    @property
    def temperature_scheme(self):
        scheme = []
        for mc in self.mc_objs:
            scheme.append(mc.T)
        return scheme

    def _init_temperature_scheme_from_file(self):
        """Initialize temperature scheme from file."""
        try:
            data = np.loadtxt(self.temperature_schedule_fname, delimiter=',')
            data = data[:, 0].tolist()
        except IOError:
            return False

        for T in data:
            new_mc = self.mc_objs[-1].copy()
            new_mc.reset()
            new_mc.T = T
            self.mc_objs.append(new_mc)
        return True

    def _adapt_temperature_schedule_to_mpi_size(self):
        """Make sure that the number of temperatures simulated match."""
        if self.mpicomm is None:
            return
        size = self.mpicomm.Get_size()
        num_temps = len(self.mc_objs)
        if num_temps % size == 0:
            return

        num_per_proc = int(num_temps/size)
        new_num_temps = size * (num_per_proc + 1)
        num_to_insert = new_num_temps - num_temps
        assert num_to_insert < num_temps

        temp_scheme = self.temperatrue_scheme
        for i in range(num_to_insert):
            new_temp = 0.5 * (temp_scheme[-1-i] + temp_scheme[-2-i])
            new_obj = self.mc_objs[-1-i].copy()
            new_obj.T = new_temp
            self.mc_objs.insert(-1-i, new_obj)
        temp_scheme = self.temperature_scheme

        # Make sure that the order is correct
        for i in range(1, len(temp_scheme)):
            assert temp_scheme[i] < temp_scheme[i-1]

        assert len(temp_scheme) % size == 0

        self._log("Requested to run with {} processors.\n"
                  "Insert more temperatures to equal load on all processors\n"
                  "Original number of temperatures: {}\n"
                  "New number of temperatures: {}\n"
                  "New temperature scheme: {}\n"
                  "".format(size, num_temps, len(temp_scheme), temp_scheme))

    def _init_temperature_scheme(self, target_accept=0.2):
        """Initialize the temperature scheme."""

        if self._init_temperature_scheme_from_file():
            # Temperature scheme was initialized from file
            self._log("Temperature schedule was initialized from file {}"
                      "".format(self.temperature_schedule_fname))
            return

        lowest_T = self.Tmax
        replica_count = 1
        self._log("Initializing temperature scheme")
        self.mc_objs[0].runMC(steps=10*self.natoms, equil=False)

        acceptance_ratios = [0.0]
        while lowest_T > self.Tmin:
            current_mc_obj = self.mc_objs[-1]
            self._log("{}: Temperature: {}K".format(replica_count,
                                                    current_mc_obj.T))
            new_mc, new_accept = self._find_next_temperature(
                current_mc_obj, target_accept=target_accept)
            if new_mc is None:
                break
            lowest_T = new_mc.T
            acceptance_ratios.append(new_accept)
            self.mc_objs.append(new_mc)
            replica_count += 1
        self._log("Temperature scheme initialized...")

        # Save temperature scheme
        temps = self.temperature_scheme
        data = np.vstack((temps, acceptance_ratios)).T
        np.savetxt(self.temperature_schedule_fname, data, delimiter=",",
                   header="Temperature (K), Acceptance probabability")
        self._log("Temperature scheme saved to {}"
                  "".format(self.temperature_schedule_fname))

    def _find_next_temperature(self, current_mc_obj, target_accept=0.2):
        """Find the text temperature given a target acceptance ratio."""
        nsteps = 10 * self.natoms
        current_temp = current_mc_obj.T

        trial_temp = current_temp/2.0
        accept_prob = 1.0
        new_mc_obj = current_mc_obj.copy()
        new_mc_obj.reset()
        cur_E = current_mc_obj.get_thermodynamic()["energy"]
        found_candidate_temp = False

        while accept_prob > target_accept and trial_temp > self.Tmin:
            new_mc_obj.T = trial_temp
            new_mc_obj.runMC(steps=nsteps, equil=False)
            stat = new_mc_obj.get_thermodynamic()
            E = stat["energy"]
            accept_prob = self._accept_probability(cur_E, E,
                                                   current_temp, trial_temp)
            found_candidate_temp = (accept_prob <= target_accept)
            trial_temp /= 2.0

        if not found_candidate_temp:
            return None, 0.0

        self._log("New candidate temperature: {}K. Accept rate: {}"
                  "".format(trial_temp, accept_prob))

        # Apply bisection method to refine the trial temperature
        converged = False
        upper_limit = current_temp
        lower_limit = trial_temp
        min_dT = 1E-4
        while not converged:
            new_T = 0.5 * (upper_limit + lower_limit)
            new_mc_obj.T = new_T
            new_mc_obj.runMC(steps=nsteps, equil=False)
            new_E = new_mc_obj.get_thermodynamic()["energy"]
            new_accept = self._accept_probability(cur_E, new_E,
                                                  current_temp, new_T)
            print(new_accept)
            if new_accept > target_accept:
                # Temperatures are too close --> lower the upper limit
                upper_limit = new_T
            else:
                # Temperature are too far --> increase the lower limit
                lower_limit = new_T

            if abs(new_accept - target_accept) < 0.01:
                converged = True
            elif upper_limit - lower_limit < min_dT:
                converged = True
        return new_mc_obj, new_accept

    def _accept_probability(self, E1, E2, T1, T2):
        """Return the acceptance probability."""
        dE = E1 - E2
        b1 = 1.0 / (kB * T1)
        b2 = 1.0 / (kB * T2)
        db = b1 - b2
        return np.exp(db * dE)

    def _exchange_configuration(self, mc1, mc2):
        """Exchange the configuration between two MC states."""
        symbs1 = [atom.symbol for atom in mc1.atoms]
        symbs2 = [atom.symbol for atom in mc2.atoms]
        mc1.set_symbols(symbs2)
        mc2.set_symbols(symbs1)

    def _replica_on_processor(self, rank):
        if self.mpicomm is None:
            return list(range(len(self.mc_objs)))

        num_proc = self.mpicomm.Get_size()
        num_per_proc = len(self.mc_objs/num_proc)
        start = self.rank * num_proc
        end = self.rank + num_per_proc
        return list(range(start, end))

    def _sync_states_responsible_proc(self):
        """Synchronize the atom configuration on all processors."""
        if self.mpicomm is None:
            return
        self.mpicomm.barrier()
        size = self.mpicomm.Get_size()
        for root in range(size):
            replica = self._replica_on_processor(root)
            for repl in replica:
                symbs = [atom.symb for atom in self.mc_objs[repl].atoms]
                symbs = self.mpicomm.bcast(symbs, root=root)
                self.mc_objs[repl].set_symbols(symbs)

    def _sync_states_with_master(self):
        """Synchronize nice the configurations with master."""
        if self.mpicomm is None:
            return

        self.mpicomm.barrier()
        for mc in self.mc_objs:
            symbs = [atom.symbol for atom in mc.atoms]
            symbs = self.mpicomm.bcast(symbs, root=0)
            mc.set_symbols(symbs)

    def _perform_exchange_move(self, direction="up"):
        """Peform exchange moves."""
        if direction == "up":
            moves = [(i, i+1) for i in range(0, len(self.mc_objs)-1, 2)]
        else:
            moves = [(i, i-1) for i in range(len(self.mc_objs)-1, 0, -2)]

        if self.rank == 0:
            num_accept = 0
            for move in moves:
                E1 = self.mc_objs[move[0]].current_energy_without_vib()
                E2 = self.mc_objs[move[1]].current_energy_without_vib()
                T1 = self.mc_objs[move[0]].T
                T2 = self.mc_objs[move[1]].T
                acc_prob = self._accept_probability(E1, E2, T1, T2)
                acc = np.random.rand() < acc_prob

                if acc:
                    self._exchange_configuration(self.mc_objs[move[0]],
                                                 self.mc_objs[move[1]])
                    num_accept += 1
            self._log("Number of accepted exchange moves: {} ({} %)"
                      "".format(num_accept,
                                float(100*num_accept)/len(moves)))
        self._sync_states_with_master()

    def run(self, mc_args={}, num_exchange_cycles=10):
        """Run Parallel Tempering

        :param mc_args: Dictionary with arguments to the run method
                        of the corresponding Monte Carlo object
        :param num_exchange_cycles: How many times should replica exchange
                                    be attempted
        """
        from random import choice
        exchange_move_dir = ["up", "down"]
        for replica_ech_cycle in range(num_exchange_cycles):
            for indx in self._replica_on_processor(self.rank):
                self.active_replica = indx
                self.mc_objs[indx].runMC(**mc_args)
            self._sync_states_responsible_proc()
            self._perform_exchange_move(direction=choice(exchange_move_dir))
