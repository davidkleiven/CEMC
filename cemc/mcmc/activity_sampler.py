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
            self.activity_sampler.num_computed_moves[key] += 1


class ActivitySampler(Montecarlo):
    def __init__(self, atoms, temp, **kwargs):
        self.insertion_moves = kwargs.pop("moves")
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
        self.check_user_arguments()

        self.averager_track = {}
        self.num_possible_moves = {}
        self.num_computed_moves = {}
        self.singlet_changes = {}
        at_count = self.count_atoms()
        self.current_move_type = "regular"
        self.current_move = None
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            self.averager_track[key] = 0.0
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

    def check_user_arguments(self):
        """
        Verify that the move argument given by the user is valid
        """
        ref_atom = self.insertion_moves[0][0]
        for move in self.insertion_moves:
            if (move[0] != ref_atom):
                msg = "All insertions moves needs to have the same reference\n"
                msg += "atom! That is the first element has to be the same "
                msg += "in all trial moves\n."
                msg += "Given: {}".format(self.insertion_moves)
                raise ValueError(msg)

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

        keys = self.averager_track.keys()
        # Broadcast keys from master process to ensure that all processes
        # put the values in the same order
        keys = self.mpicomm.bcast(keys, root=0)
        num_computed = np.zeros(len(keys))
        averages = np.zeros(len(keys))
        for i, key in enumerate(keys):
            averages[i] = self.averager_track[key]
            num_computed[i] = self.num_computed_moves[key]

        recv_buf = np.zeros_like(averages)
        self.mpicomm.Allreduce(averages, recv_buf)
        averages[:] = recv_buf
        self.mpicomm.Allreduce(num_computed, recv_buf)
        num_computed[:] = recv_buf

        # Distribute back into the original datastructures
        for i, key in enumerate(keys):
            self.averager_track[key] = averages[i]
            self.num_computed_moves[key] = num_computed[i]

    def get_thermodynamic(self):
        """
        Override the thermodynamics function
        """
        self.collect_results()
        res = {}
        res = Montecarlo.get_thermodynamic(self)

        # Inser the insertion enegies
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            name = "insert_energy_{}".format(key)
            res[name] = self.averager_track[key] / \
                self.num_computed_moves[key]
        return res
