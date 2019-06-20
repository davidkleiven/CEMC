from cemc.mcmc import Montecarlo
from cemc.mcmc.mc_observers import MCObserver
from cemc.mcmc.exponential_weighted_averager import ExponentialWeightedAverager
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

            self.activity_sampler.averager_track[key].add(dE)
            self.activity_sampler.raw_insertion_energy[key] += dE
            self.activity_sampler.boltzmann_weight_ins_energy[key].add(dE)
            self.activity_sampler.boltzmann_weight_ins_eng_sq[key].add(dE)


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

        # Filter out impossible moves
        self.insertion_moves = self._filter_by_possible_moves()

        self.averager_track = {}
        self.raw_insertion_energy = {}
        self.boltzmann_weight_ins_energy = {}
        self.boltzmann_weight_ins_eng_sq = {}
        self.num_possible_moves = {}
        self.singlet_changes = {}
        at_count = self.count_atoms()
        self.current_move_type = "regular"
        self.current_move = None
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            self.averager_track[key] = \
                ExponentialWeightedAverager(self.T, order=0)

            self.raw_insertion_energy[key] = 0.0
            self.boltzmann_weight_ins_energy[key] = \
                ExponentialWeightedAverager(self.T, order=1)
            self.boltzmann_weight_ins_eng_sq[key] = \
                ExponentialWeightedAverager(self.T, order=2)
            self.num_possible_moves[key] = at_count[move[0]]
            self.singlet_changes[key] = []

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

    def _filter_by_possible_moves(self):
        """Filter out the moves that are not possible due to missing atoms.

        :return: List with filtered moves
        """
        filtered_moves = []
        at_count = self.count_atoms()
        for move in self.insertion_moves:
            if move[0] in at_count.keys():
                filtered_moves.append(move)
        return filtered_moves

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

    def reset(self):
        """
        Rests all sampled data
        """
        super(ActivitySampler, self).reset()

        for key in self.averager_track.keys():
            self.averager_track[key].reset()
            self.raw_insertion_energy[key] = 0.0
            self.boltzmann_weight_ins_energy[key].reset()
            self.boltzmann_weight_ins_eng_sq[key].reset()
        self.current_move_type = "regular"
        self.current_move = None

    def get_key(self, move_from, move_to):
        """
        Returns the key to the dictionary

        :return: str with name of the move
        """
        return "{}to{}".format(move_from, move_to)

    def _get_trial_move(self):
        """
        Override the parents trial move class

        :return: list of tuples describing the trial move
        """
        if (np.random.rand() < self.prob_insert_move):
            self.current_move_type = "insert_move"
            # Try to introduce another atom
            move = self.insertion_moves[np.random.randint(
                low=0, high=len(self.insertion_moves))]
            # indices = self.atoms_indx[move[0]]
            # indx = indices[np.random.randint(low=0, high=len(indices))]
            indx = self.atoms_tracker.get_random_indx_of_symbol(move[0])
            system_changes = [(indx, move[0], move[1])]
            self.current_move = self.get_key(move[0], move[1])
            return system_changes
        else:
            self.current_move_type = "regular"
            return super(ActivitySampler, self)._get_trial_move()

    def _accept(self, system_changes):
        """
        Override parents accept function

        :return: True/False if True the move will be accepted
        """
        move_accepted = super(ActivitySampler, self)._accept(system_changes)
        if (self.current_move_type == "insert_move"):
            # Always reject such that the composition is conserved.
            # The new_energy will however be updated so we can use this
            return False
        return move_accepted

    def get_thermodynamic(self):
        """
        Override the thermodynamics function.

        Normalization of the averagers:
        The values in average track should represent the ratio
        between partition functions when one have one extra of the
        inserted atom and one less of the removed atom.
        In the high temperature limit this is given by

        Z_{n+1} = N!/((n+1)!(N-n-1)!)

        and

        Z_n = N!/(n!(N-n)!)

        where N is the overall number of atoms of the two atoms being swapped.
        Hence, the ratio becomes

        Z_{n+1}/Z_n = (N-n)/(n+1),

        the sums are normalized such that they results in this value
        at high temperatures.

        :return: dict with the thermodynamical properties
        """
        res = {}
        res = Montecarlo.get_thermodynamic(self)
        at_count = self.count_atoms()

        # Inser the insertion enegies
        for move in self.insertion_moves:
            key = self.get_key(move[0], move[1])
            name = "insert_energy_{}".format(key)
            N = self.averager_track[key].num_samples
            inf_temp = float(at_count[move[0]])/(at_count[move[1]]+1)
            E0 = self.averager_track[key].ref_value

            res[name] = E0 - kB * self.T * \
                np.log(self.averager_track[key].average * inf_temp)

            name = "raw_insert_energy_{}".format(key)
            res[name] = self.raw_insertion_energy[key] / N

            name = "boltzmann_avg_insert_energy_{}".format(key)
            res[name] = self.boltzmann_weight_ins_energy[key].average / \
                self.averager_track[key].average

            name = "boltzmann_avg_insert_energy_sq_{}".format(key)
            res[name] = self.boltzmann_weight_ins_eng_sq[key].average / \
                self.averager_track[key].average
        return res
