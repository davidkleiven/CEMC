from cemc.mcmc import NucleationMC
import json
import numpy as np
from ase.units import kB

class TransitionPathRelaxer(object):
    def __init__( self, nuc_mc=None ):
        if ( not isinstance(nuc_mc,NucleationMC) ):
            raise TypeError( "nuc_mc has to be an instance of NucleationMC" )
        self.nuc_mc = nuc_mc
        self.nuc_mc.set_mode( "transition_path_sampling" )
        self.init_path = None
        self.initial_peak_energy = None

    def load_path( self, fname ):
        """
        Loads the initial path
        """
        with open(fname,'r') as infile:
            self.init_path = json.load(infile)
        self.initial_peak_energy = np.max(self.init_path["energy"])
        self.nuc_mc.min_size_product = self.init_path["min_size_product"]
        self.nuc_mc.max_size_reactant = self.init_path["max_size_reactant"]

    def shooting_move( self, timeslice, direction="forward" ):
        """
        Performs one shooting move
        """
        if ( direction == "forward" ):
            stepsize = 1
            endtime = len(self.init_path["energy"])
            start = timeslice+1
        elif ( direction == "backward" ):
            stepsize = -1
            endtime = 0
            start = timeslice-1

        new_path = {}
        new_path["symbols"] = []
        new_path["energy"] = []
        self.nuc_mc.set_state( self.init_path["symbols"][timeslice] )
        self.nuc_mc.current_energy = self.init_path["energy"][timeslice]
        for i in range(start,endtime,stepsize):
            self.nuc_mc.sweep()
            E_new = self.nuc_mc.current_energy
            E_old = self.init_path["energy"][i]
            if ( self.accept(E_new,E_old) ):
                new_path["energy"].append(E_new)
                new_path["symbols"].append( [atom.symbol for atom in self.nuc_mc.atoms] )
            else:
                return

        self.nuc_mc.network.reset()
        self.nuc_mc.network(None) # Construct the network to to check the endpoints

        # Figure out if the system ended up in the correct target
        ended_in_correct_basin = False
        if ( direction == "forward" ):
            if ( self.nuc_mc.is_product() ):
                ended_in_correct_basin = True
        elif ( direction == "backward" ):
            if ( self.nuc_mc.is_reactant() ):
                ended_in_correct_basin = True

        path_length = np.abs(endtime-timeslice)
        if ( ended_in_correct_basin ):
            self.log ("New {} path accepted. Path length {}".format(direction,path_length) )
            if ( direction == "forward" ):
                self.init_path["energy"][start:] = new_path["energy"]
                self.init_path["symbols"][start:] = new_path["symbols"]
            elif ( direction == "backward" ):
                self.init_path["energy"][:timeslice] = new_path["energy"]
                self.init_path["symbols"][:timeslice] = new_path["symbols"]
        else:
            self.log( "New path rejected because it ended in the wrong basin. Path length {}".format(path_length))

    def accept( self, E_new, E_old ):
        """
        Check if the move is accepted
        """
        kT = kB*self.nuc_mc.T
        diff = E_new-E_old
        if ( diff < 0.0 ):
            return True
        return np.random.rand() < np.exp(-diff/kT )

    def log(self, msg ):
        """
        Log result to a file
        """
        print(msg)

    def relax_path( self, initial_path=None, n_shooting_moves=10000 ):
        """
        Relax the transiton path by performing shooting moves
        """
        if ( initial_path is None ):
            raise ValueError( "Filename containing the initial path was not given!" )
        self.load_path( initial_path )
        self.nuc_mc.remove_network_observers()

        for move in range(n_shooting_moves):
            self.nuc_mc.reset()
            self.log( "Move {} of {}".format(move,n_shooting_moves) )
            timeslice = np.random.randint(low=0,high=len(self.init_path["energy"]) )
            self.shooting_move(timeslice,direction="forward")
            self.nuc_mc.reset()
            self.shooting_move(timeslice,direction="backward")

        ofname = initial_path.rpartition(".")[0]
        ofname += "_relaxed.json"
        with open(ofname,'w') as outfile:
            json.dump(self.init_path,outfile)
        self.log( "Relaxed path written to {}".format(ofname) )
        new_peak_energy = np.max(self.init_path["energy"])
        self.log( "Maximum energy along path changed from {} eV to {} eV".format(self.initial_peak_energy,new_peak_energy) )
