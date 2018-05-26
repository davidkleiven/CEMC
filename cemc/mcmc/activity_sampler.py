from montecarlo import Montecarlo
from itertools import combinations
from mc_observers import MCObserver
from ase.units import kB
import numpy as np
import json
import dataset

class TrialEnergyObserver( MCObserver ):
    def __init__( self, activity_sampler ):
        self.activity_sampler = activity_sampler
        self.name = "TrialEnergyObserver"

    def __call__( self, system_changes ):
        if ( self.activity_sampler.current_move_type == "insert_move" ):
            key = self.activity_sampler.current_move
            dE = self.activity_sampler.new_energy - self.activity_sampler.current_energy
            interaction_energy = dE - self.activity_sampler.eci_singlets.dot(self.activity_sampler.singlet_changes[key])
            beta = 1.0/(kB*self.activity_sampler.T)
            self.activity_sampler.averager_track[key] += np.exp(-beta*interaction_energy)
            self.activity_sampler.num_computed_moves[key] += 1


class ActivitySampler( Montecarlo ):
    def __init__( self, atoms, temp, **kwargs ):
        self.insertion_moves = kwargs.pop("moves")
        self.symbols = []
        for move in self.insertion_moves:
            if ( len(move) != 2 ):
                msg = "The format of the insertion moves appears to be wrong\n"
                msg += "Example: [(Al,Mg),(Al,Si),(Al,Cu)]\n"
                msg += "Given: {}".format(self.insertion_moves)
                raise ValueError(msg)

            for symb in move:
                if ( symb not in self.symbols ):
                    self.symbols.append(symb)

        self.prob_insert_move = 0.1
        if ( "prob_insert_move" in kwargs.keys() ):
            self.prob_insert_move = kwargs.pop("prob_insert_move")

        super(ActivitySampler,self).__init__( atoms, temp, **kwargs)
        self.check_user_arguments()

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

        self.eci_singlets = np.zeros(len(self.symbols)-1)
        self.find_singlet_changes()
        self.required_tables = None
        self.required_fields = None

        # Observer that tracks the energy of the trial move
        self.trial_energy_obs = TrialEnergyObserver(self)
        self.attach( self.trial_energy_obs )

    def check_user_arguments( self ):
        """
        Verify that the move argument given by the user is valid
        """
        ref_atom = self.insertion_moves[0][0]
        for move in self.insertion_moves:
            if ( move[0] != ref_atom ):
                msg = "All insertions moves needs to have the same reference atom!\n"
                msg += "That is the first element has to be the same in all trial moves\n"
                msg += "Given: {}".format(self.insertion_moves)
                raise ValueError(msg)

        # Check that all symbols are present in the atoms object
        at_count = self.count_atoms()
        if ( len(at_count.keys()) != len(self.insertion_moves)+1 ):
            n_atoms = len(at_count.keys())
            msg =  "Detected {} atoms. Then the number of moves has to be {}\n".format(n_atoms,n_atoms-1)
            msg += "Moves given: {}".format(self.insertion_moves)
            raise ValueError(msg)

    def find_singlet_changes(self):
        """
        Try all possible insertion moves and find out how much the singlets changes
        """
        bfs = self.atoms._calc.BC.basis_functions
        for move in self.insertion_moves:
            key = self.get_key( move[0], move[1] )
            for bf in bfs:
                self.singlet_changes[key].append( bf[move[1]] - bf[move[0]] )
            self.singlet_changes[key] = np.array(self.singlet_changes[key])

        # Update the corresponding ECIs
        for name, value in self.atoms._calc.eci.iteritems():
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
        return "{}to{}".format(move_from,move_to)

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
            indices = self.atoms_indx[move[0]]
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
        move_accepted = super(ActivitySampler,self).accept(system_changes)
        if ( self.current_move_type == "insert_move" ):
            # Always reject such that the composition is conserved.
            # The new_energy will however be updated so we can use this
            return False
        return move_accepted

    def collect_results(self):
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
        self.mpicomm.Allreduce( averages, recv_buf )
        averages[:] = recv_buf
        self.mpicomm.Allreduce( num_computed, recv_buf )
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
        res["effective_conc"] = {}
        res["temperature"] = self.T

        at_count = self.count_atoms()
        concs = {key:0.0 for key in self.symbols}
        for key,value in at_count.iteritems():
            concs[key] = float(value)/len(self.atoms)

        beta = 1.0/(kB*self.T)
        n_elem = len(self.insertion_moves)

        # Matrix for solving the effective concentrations
        A = np.zeros((n_elem,n_elem))
        rhs = np.zeros(n_elem)

        # Dictionaray to relate each element to a row/column in the matrix
        row = {move[1]:i for i,move in enumerate(self.insertion_moves)}
        for move in self.insertion_moves:
            key = self.get_key( move[0], move[1] )
            normalization = float(at_count[move[1]]+1)/at_count[move[0]]
            res["activity_coefficient"][key] = self.averager_track[key]/self.num_computed_moves[key]
            res["activity"][key] = res["activity_coefficient"][key]*normalization

            # Build matrix
            y = res["activity"][key]
            cur_row = row[move[1]]
            A[cur_row,:] = y
            A[cur_row,cur_row] = 1.0+y
            rhs[cur_row] = y*len(self.atoms)-1

        eff_conc = np.linalg.solve(A,rhs)/len(self.atoms)
        for i in range(len(eff_conc)):
            res["effective_conc"][self.insertion_moves[i][1]] = eff_conc[i]

        res["effective_conc"][self.insertion_moves[0][0]] = 1.0-np.sum(eff_conc)
        res["conc"] = concs
        return res

    def save( self, fname="default.db" ):
        """
        Store the results into an ASE database
        """
        res = self.get_thermodynamic()

        if ( self.rank == 0 ):
            db = dataset.connect(self.db_url(fname))
            nproc = 1
            if ( self.mpicomm is not None ):
                nproc = self.mpicomm.Get_size()
            name = "{}K_{}".format(int(self.T),self.atoms.get_chemical_formula())
            systems = db["systems"]
            entry = systems.find_one(name=name)
            if ( entry is not None ):
                # Entry already exists
                N_prev = entry["n_mc_steps"]
                nsteps = self.current_step*nproc
                sysID = entry["id"]
                activity_table = db["activity"]
                activity = activity_table.find_one(sysID=sysID)
                for key in res["activity"].keys():
                    activity[key] = (activity[key]*N_prev + res["activity"][key]*nsteps)/(N_prev+nsteps)
                activity_table.update( activity, ["sysID"] )

                act_coeff_table = db["activity_coefficient"]
                act_coeff = act_coeff_table.find_one(sysID=sysID)
                for key in res["activity_coefficient"].keys():
                    act_coeff[key] = (act_coeff[key]*N_prev + res["activity_coefficient"][key]*nsteps)/(N_prev+nsteps)
                act_coeff_table.update( act_coeff, ["sysID"])

                eff_conc_tab = db["effective_conc"]
                eff_conc = eff_conc_tab.find_one(sysID=sysID)
                for key in res["effective_conc"].keys():
                    eff_conc[key] = (eff_conc[key]*N_prev + nsteps*res["effective_conc"][key])/(N_prev+nsteps)
                entry["n_mc_steps"] = N_prev+nsteps
                systems.update( energy, ["sysID"])
            else:
                at_count = self.count_atoms()
                conc = {}
                for key in at_count.keys():
                    conc[key] = float(at_count[key])/len(self.atoms)
                uid = systems.insert(dict(temperature=int(self.T),n_mc_steps=self.current_step*nproc,\
                formula=self.atoms.get_chemical_formula()))

                activity_tab = db["activity"]
                res["activity"].update({"sysID":uid})
                activity_tab.insert( res["activity"] )

                activity_coefficient = db["activity_coefficient"]
                res["activity_coefficient"].update({"sysID":uid})
                activity_coefficient.insert( res["activity_coefficient"])

                eff_conc = db["effective_conc"]
                res["effective_conc"].update({"sysID":uid})
                eff_conc.insert( res["effective_conc"] )
                conc = db["concentration"]
                res["conc"].update({"sysID":uid})
                conc.insert( res["conc"] )
        self.log( "Results written to {}".format(fname) )

    @staticmethod
    def db_url( fname ):
        return "sqlite:///"+fname

    @staticmethod
    def effective_composition(db_name,temperature):
        """
        Function included for convenience to get datastructures that can
        easily be plotted
        """
        db = dataset.connect( ActivitySampler.db_url(db_name) )

        res = {"conc":{},"eff_conc":{}}
        systems = db["systems"]
        fetched = systems.find(temperature=temperature)
        if ( fetched is None ):
            raise ValueError("No entries in the data base with temperature {}K".format(temperature))

        for entry in fetched:
            sysID = entry["id"]
            conc = db["concentration"].find_one(sysID=sysID)
            eff_conc = db["effective_conc"].find_one(sysID=sysID)
            conc.pop("sysID")
            eff_conc.pop("sysID")
            conc.pop("id")
            eff_conc.pop("id")
            for key in conc.keys():
                if ( key not in res["conc"].keys()):
                    res["conc"][key] = []
                    res["eff_conc"][key] = []
                res["conc"][key].append( conc[key] )
                res["eff_conc"][key].append( eff_conc[key] )

        # Sort the data
        for key in res["conc"].keys():
            srt_indx = np.argsort(res["conc"][key])
            res["conc"][key] = [res["conc"][key][indx] for indx in srt_indx]
            res["eff_conc"][key] = [res["eff_conc"][key][indx] for indx in srt_indx]
        return res
