from cemc.mcmc import Montecarlo
from cemc.mcmc.mc_observers import MCObserver
from ase.units import kB
import numpy as np


class TrialEnergyObserver(MCObserver):
    def __init__(self, activity_sampler):
        self.activity_sampler = activity_sampler
        self.name = "TrialEnergyObserver"

    def __call__(self, system_changes):
        if (self.activity_sampler.current_move_type == "insert_move"):
            key = self.activity_sampler.current_move
            dE = self.activity_sampler.new_energy - \
                self.activity_sampler.current_energy
            beta = 1.0 / (kB * self.activity_sampler.T)
            self.activity_sampler.averager_track[key] += np.exp(-beta * dE)

            self.activity_sampler.raw_insertion_energy[key] += dE

            self.activity_sampler.boltzmann_weight_ins_energy[key] += \
                dE*np.exp(-beta * dE)

            self.activity_sampler.boltzmann_weight_ins_eng_eq[key] += \
                dE**2 * np.exp(-beta * dE)

            self.activity_sampler.num_computed_moves[key] += 1


class ActivitySampler(Montecarlo):
    def __init__(self, atoms, temp, **kwargs):
        self.insertion_moves = kwargs.pop("moves")
        self._expand_moves_with_inverse()
        self.symbols = []
        for move in self.insertion_moves:
            if (len(move) != 2):
                msg = "The format of the insertion moves appears to be wrong\n"
                msg += "Example: [(Al,Mg),(Al,Si),(Al,Cu)]\n"
                msg += "Given: {}".format(self.insertion_moves)
                raise ValueError(msg)

            for symb in move:
                if (symb not in self.symbols):
                    self.symbols.append(symb)

        self.prob_insert_move = 1.0 / len(atoms)
        if ("prob_insert_move" in kwargs.keys()):
            self.prob_insert_move = kwargs.pop("prob_insert_move")

        super(ActivitySampler, self).__init__(atoms, temp, **kwargs)

        self.averager_track = {}
        self.raw_insertion_energy = {}
        self.boltzmann_weight_ins_energy = {}
        self.boltzmann_weight_ins_eng_eq = {}
        self.num_possible_moves = {}
        self.num_computed_moves = {}
        self.singlet_changes = {}
        at_count = self.count_atoms()
        self.current_move_type = "regular"
        self.current_move = None
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            self.averager_track[key] = 0.0
            self.raw_insertion_energy[key] = 0.0
            self.boltzmann_weight_ins_energy[key] = 0.0
            self.boltzmann_weight_ins_eng_eq[key] = 0.0
            self.num_possible_moves[key] = at_count[move[0]]
            self.num_computed_moves[key] = 0
            self.singlet_changes[key] = []

        self.eci_singlets = np.zeros(len(self.symbols) - 1)
        self.find_singlet_changes()
        self.required_tables = None
        self.required_fields = None

        # Observer that tracks the energy of the trial move
        self.trial_energy_obs = TrialEnergyObserver(self)
        self.attach(self.trial_energy_obs)
        self.log("===========================================================")
        self.log("==   WARNING! THIS CODE IS NOT PROPERLY TESTED. ASK THE  ==")
        self.log("==                DEVELOPERS BEFORE USE                  ==")
        self.log("===========================================================")

    def _expand_moves_with_inverse(self):
        """Expand all the moves given by the user with the inverse moves."""
        new_moves = []
        for move in self.insertion_moves:
            new_moves.append((move[1], move[0]))
        self.insertion_moves += new_moves

    def find_singlet_changes(self):
        """
        Try all possible insertion moves and find the singlet change
        """
        bfs = self.atoms._calc.BC.basis_functions
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            for bf in bfs:
                self.singlet_changes[key].append(bf[move[1]] - bf[move[0]])
            self.singlet_changes[key] = np.array(self.singlet_changes[key])

        # Update the corresponding ECIs
        for name, value in self.atoms._calc.eci.items():
            if (name.startswith("c1")):
                self.eci_singlets[int(name[-1])] = value

    def reset(self):
        """
        Rests all sampled data
        """
        super(ActivitySampler, self).reset()

        for key in self.averager_track.keys():
            self.averager_track[key] = 0.0
            self.raw_insertion_energy[key] = 0.0
            self.boltzmann_weight_ins_energy[key] = 0.0
            self.boltzmann_weight_ins_eng_eq[key] = 0.0
            self.num_computed_moves[key] = 0
        self.current_move_type = "regular"
        self.current_move = None

    def get_key(self, move_from, move_to):
        """
        Returns the key to the dictionary
        """
        return "{}to{}".format(move_from, move_to)

    def _get_trial_move(self):
        """
        Override the parents trial move class
        """
        if (np.random.rand() < self.prob_insert_move):
            self.current_move_type = "insert_move"
            # Try to introduce another atom
            move = self.insertion_moves[np.random.randint(
                low=0, high=len(self.insertion_moves))]
            indices = self.atoms_indx[move[0]]
            indx = indices[np.random.randint(low=0, high=len(indices))]
            system_changes = [(indx, move[0], move[1])]
            self.current_move = self.get_key(move[0], move[1])
            return system_changes
        else:
            self.current_move_type = "regular"
            return super(ActivitySampler, self)._get_trial_move()

    def _accept(self, system_changes):
        """
        Override parents accept function
        """
        move_accepted = super(ActivitySampler, self)._accept(system_changes)
        if (self.current_move_type == "insert_move"):
            # Always reject such that the composition is conserved.
            # The new_energy will however be updated so we can use this
            return False
        return move_accepted

    def collect_results(self):
        """
        Collect results from all processors
        """
        if (self.mpicomm is None):
            return

        keys = list(self.averager_track.keys())
        # Broadcast keys from master process to ensure that all processes
        # put the values in the same order
        keys = self.mpicomm.bcast(keys, root=0)
        num_computed = np.zeros(len(keys))
        averages = np.zeros(len(keys))
        raw_energies = np.zeros_like(averages)
        boltzmann_eng = np.zeros_like(averages)
        boltzmann_eng_sq = np.zeros_like(averages)
        for i, key in enumerate(keys):
            averages[i] = self.averager_track[key]
            num_computed[i] = self.num_computed_moves[key]
            raw_energies[i] = self.raw_insertion_energy[key]
            boltzmann_eng[i] = self.boltzmann_weight_ins_energy[key]
            boltzmann_eng_sq[i] = self.boltzmann_weight_ins_eng_eq[key]

        recv_buf = np.zeros_like(averages)
        self.mpicomm.Allreduce(averages, recv_buf)
        averages[:] = recv_buf

        self.mpicomm.Allreduce(num_computed, recv_buf)
        num_computed[:] = recv_buf

        self.mpicomm.Allreduce(raw_energies, recv_buf)
        raw_energies[:] = recv_buf

        self.mpicomm.Allreduce(boltzmann_eng, recv_buf)
        boltzmann_eng[:] = recv_buf

        self.mpicomm.Allreduce(boltzmann_eng_sq, recv_buf)
        boltzmann_eng_sq[:] = recv_buf

        # Distribute back into the original datastructures
        for i, key in enumerate(keys):
            self.averager_track[key] = averages[i]
            self.num_computed_moves[key] = num_computed[i]
            self.raw_insertion_energy[key] = raw_energies[i]
            self.boltzmann_weight_ins_energy[key] = boltzmann_eng[i]
            self.boltzmann_weight_ins_eng_eq[key] = boltzmann_eng_sq[i]

    def get_thermodynamic(self):
        """
        Override the thermodynamics function
        """
        self.collect_results()
        res = {}
        res = Montecarlo.get_thermodynamic(self)
        at_count = self.count_atoms()

        # Inser the insertion enegies
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            name = "insert_energy_{}".format(key)
            N = self.num_computed_moves[key]
            res[name] = self.averager_track[key] * at_count[move[0]] / N

            name = "raw_insert_energy_{}".format(key)
            res[name] = self.raw_insertion_energy[key] / N

            name = "boltzmann_avg_insert_energy_{}".format(key)
            res[name] = self.boltzmann_weight_ins_energy[key] / \
                self.averager_track[key]

            name = "boltzmann_avg_insert_energy_sq_{}".format(key)
            res[name] = self.boltzmann_weight_ins_eng_eq[key] / \
                self.averager_track[key]
        return res
