from montecarlo import Montecarlo
from itertools import combinations
from mc_observers import MCObserver
from ase.units import kB

class TrialEnergyObserver( MCObserver ):
    def __init__( self, activity_sampler ):
        self.activity_sampler = activity_sampler
        self.name = "TrialEnergyObserver"

    def __call__( self, system_changes ):
        if ( self.activity_sampler.current_move_type == "insert_move" ):
            key = self.activity_sampler.current_move
            dE = self.activity_sampler.new_energy - self.activity_sampler.current_energy
            interaction_energy = dE - self.activity_sampler.eci_singlets.dot(self.singlet_changes[key])
            beta = 1.0/(kB*self.activity_sampler.T)
            self.activity_sampler.averager_track[key] += np.exp(-beta*interaction_energy)
            self.activity_sampler.num_computed_moves[key] += 1


class ActivitySampler( Montecarlo ):
    def __init__( self, atoms, temp, **kwargs ):
        self.symbols = kwargs.pop("symbols")
        self.prob_insert_move = 0.1
        if ( "prob_insert_move" in kwargs.keys() ):
            self.prob_insert_move = kwargs.pop("prob_insert_move")

        super(ActivitySampler,self).__init__(**kwargs)
        self.insertion_moves = combinations(self.symbols,2)
        self.averager_track = {}
        self.num_possible_moves = {}
        self.num_computed_moves = {}
        self.singlet_changes  = {}
        at_count = self.count_atoms()
        self.current_move_type = "regular"
        self.current_move = None
        for move in self.insertion_moves:
            key = self.get_key( move[0], move[1] )
            self.averager_track[key] = 0.0
            self.num_possible_moves[key] = at_count[move[0]]
            self.num_computed_moves[key] = 0
            self.singlet_changes[key] = []
        self.trial_energy_obs = TrialEnergyObserver(self)
        self.attach( self.trial_energy_obs )
        self.find_singlet_changes()
        self.eci_singlets = np.zeros(len(self.symbols)-1)

    def find_singlet_changes(self):
        """
        Try all possible insertion moves and find out how much the singlets changes
        """
        bfs = self.atoms._calc.BC.basis_funcitons
        for move in self.insertion_moves:
            key = self.get_key( move[0], move[1] )
            for bf in bfs:
                singlet_changes[key].append( bf[move[1]] - bf[move[0]] )
            self.singlet_changes[key] = np.array(self.singlet_changes[key])

        # Update the corresponding ECIs
        for name, value in self.atoms._calc.ecis.iteritems():
            if ( name.startswith("c1") ):
                self.eci_singlets[int(name[-1])] = value

    def reset(self):
        """
        Rests all sampled data
        """
        super(ActivitySampler,self).reset()

        for key in self.averager_track.keys():
            self.averager_track[key] = 0.0
            self.num_computed_moves[key] = 0
        self.current_move_type = "regular"
        self.current_move = None

    def get_key( self, move_from, move_to ):
        """
        Returns the key to the dictionary
        """
        return "{}->{}".format(move_from,move_to)

    def count_atoms(self):
        """
        Count the number of each species
        """
        atom_count = {key:0 for key in self.symbols}
        for atom in self.atoms:
            atom_count[atom.symbol] += 1
        return atom_count

    def get_trial_move( self ):
        """
        Override the parents trial move class
        """
        if ( np.random.rand() < self.prob_insert_move ):
            self.current_move_type = "insert_move"
            # Try to introduce another atom
            move = self.insertion_moves[np.random.randint(low=0,high=len(self.insertion_moves))]
            indices = self.atom_indx[move[0]]
            indx = indices[np.random.randint(low=0,high=len(indices))]
            system_changes = [(indx,move[0],move[1])]
            self.current_move = self.get_key( move[0], move[1] )
            return system_changes
        else:
            self.current_move_type = "regular"
            return super(ActivitySampler,self).get_trial_move()

    def accept( self, system_changes ):
        """
        Override parents accept function
        """
        if ( self.current_move_type == "insert_move" ):
            # Always reject such that the composition is conserved.
            # The new_energy will however be updated so we can use this
            return False
        return super(ActivitySampler,self).accept(system_changes)

    def collect_result(self):
        """
        Collect results from all processors
        """
        if ( self.mpicomm is None ):
            return

        keys = self.averager_track.keys()
        # Broadcast keys from master process to ensure that all processes
        # put the values in the same order
        keys = self.mpicomm.bcast(keys,root=0)
        num_computed = np.zeros(len(keys))
        averages = np.zeros(len(keys))
        for i,key in enumerate(keys):
            averages[i] = self.averager_track[key]
            num_computed[i] = self.num_computed_moves[key]

        recv_buf = np.zeros_like(averages)
        self.Allreduce( averages, recv_buf )
        averages[:] = recv_buf
        self.Allreduce( num_computed, recv_buf )
        num_computed[:] = recv_buf

        # Distribute back into the original datastructures
        for i,key in enumerate(keys):
            self.averager_track[key] = averages[i]
            self.num_computed_moves[key] = num_computed[i]


    def get_thermodynamic(self):
        """
        Override the thermodynamics function
        """
        self.collect_results()
        res = {}
        res["energy_changes"] = {}
        res["activity"] = {}
        res["activity_coefficient"] = {}

        at_count = self.count_atoms()
        concs = {key:0.0 for key in self.symbols}
        for key,value in at_count.iteritems():
            concs[key] = float(at_count)/len(self.atoms)

        beta = 1.0/(kB*self.T)
        for move in self.insertion_moves:
            key = self.get_key( move[0], move[1] )
            res["energy_changes"][key] = self.num_possible_moves[key]*self.averager_track[key]/self.num_computed_moves[key]
            res["activity"][key] = res["energy_changes"][key]
            res["activity_coefficient"][key] = res["activity"][key]/concs[move[1]]
        return res
