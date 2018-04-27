from cemc.mcmc import NucleationMC
import json
import numpy as np
from ase.units import kB
from ase.io.trajectory import TrajectoryWriter
import copy
from matplotlib import pyplot as plt

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
            n_sweeps = len(self.init_path["energy"])-timeslice+1
        elif ( direction == "backward" ):
            n_sweeps = timeslice

        new_path = {}
        new_path["symbols"] = []
        new_path["energy"] = []
        self.nuc_mc.set_state( self.init_path["symbols"][timeslice] )
        self.nuc_mc.current_energy = self.init_path["energy"][timeslice]
        for i in range(n_sweeps):
            self.nuc_mc.network.reset()
            self.nuc_mc.sweep()
            self.nuc_mc.network(None)
            new_path["energy"].append(self.nuc_mc.current_energy)
            new_path["symbols"].append( [atom.symbol for atom in self.nuc_mc.atoms] )
            if ( self.nuc_mc.is_product() and direction=="backward" ):
                self.log ("Ended up in product basin, when propagation direction is backward")
                return False
            elif ( self.nuc_mc.is_reactant() and direction=="forward" ):
                self.log("Ended up in reactant basin when propagation direction is forward")
                return False

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

        path_length = n_sweeps
        if ( ended_in_correct_basin ):
            self.log ("New {} path accepted. Path length {}".format(direction,path_length) )
            if ( direction == "forward" ):
                self.init_path["energy"][timeslice+1:] = new_path["energy"]
                self.init_path["symbols"][timeslice+1:] = new_path["symbols"]
                return True
            elif ( direction == "backward" ):
                self.init_path["energy"][:timeslice] = new_path["energy"]
                self.init_path["symbols"][:timeslice] = new_path["symbols"]
                return True
        else:
            self.log( "New path rejected because it ended in the wrong basin. Path length {}".format(path_length))
        return False

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
        min_slice,max_slice = self.find_timeslices_in_transition_region()
        self.log ( "First timeslice outside reaction region: {}".format(min_slice))
        self.log ( "First timeslice in product region: {}".format(max_slice))

        direcions = ["forward","backward"]
        for move in range(n_shooting_moves):
            self.nuc_mc.reset()
            direct = direcions[np.random.randint(low=0,high=2)]
            self.log( "Move {} of {}".format(move,n_shooting_moves) )
            timeslice = np.random.randint(low=min_slice,high=max_slice )
            self.shooting_move(timeslice,direction=direct)

        ofname = initial_path.rpartition(".")[0]
        ofname += "_relaxed.json"
        with open(ofname,'w') as outfile:
            json.dump(self.init_path,outfile)
        self.log( "Relaxed path written to {}".format(ofname) )
        new_peak_energy = np.max(self.init_path["energy"])
        self.log( "Maximum energy along path changed from {} eV to {} eV".format(self.initial_peak_energy,new_peak_energy) )

    def path2trajectory( self, fname="relaxed_path.traj" ):
        """
        Writes the path to a trajectory file
        """
        traj = TrajectoryWriter(fname,'w')
        for state in self.init_path["symbols"]:
            self.nuc_mc.network.reset()
            self.nuc_mc.set_state(state)
            self.nuc_mc.network(None)
            atoms = self.nuc_mc.network.get_atoms_with_largest_cluster( prohibited_symbols=["Al","Mg"] )
            if ( atoms is None ):
                traj.write(self.nuc_mc.atoms )
            else:
                traj.write(atoms)
        self.log( "Trajectory written to {}".format(fname))

    def find_timeslices_in_transition_region(self):
        """
        Locate the time slices in the transition region
        """
        min_slice_outside_reactant = None
        max_slice_outside_products = None
        for timeslice,state in enumerate(self.init_path["symbols"]):
            self.nuc_mc.reset()
            self.nuc_mc.set_state( state )
            self.nuc_mc.network(None)
            if ( not self.nuc_mc.is_reactant() and min_slice_outside_reactant is None ):
                min_slice_outside_reactant = timeslice
            elif ( self.nuc_mc.is_product() and max_slice_outside_products is None ):
                max_slice_outside_products = timeslice
        return min_slice_outside_reactant,max_slice_outside_products


    def generate_paths( self, initial_path=None, n_paths=1, max_attempts=10000, outfile="tse_ensemble.json" ):
        """
        Generate a given number of paths
        """
        all_paths = []
        direction = ["forward","backward"]
        self.load_path(initial_path)
        orig_path = copy.deepcopy(self.init_path)
        if ( n_paths >= max_attempts ):
            raise ValueError( "The number of paths requested exceeds the maximum number of attempts. Increase the maximum number of attempts" )
        counter = 0
        min_slice,max_slice = self.find_timeslices_in_transition_region()
        while( len(all_paths) < n_paths and counter < max_attempts ):
            counter += 1
            self.init_path = copy.deepcopy(orig_path)
            direct = direction[np.random.randint(low=0,high=2)]
            timeslice = np.random.randint(low=min_slice,high=max_slice)
            if ( self.shooting_move( timeslice, direction=direct) ):
                all_paths.append(self.init_path)
        self.save_tse_ensemble( all_paths, fname=outfile )

    def save_tse_ensemble(self, new_paths, fname="tse_ensemble.json" ):
        """
        Save the new paths to file
        """
        data = {"transition_paths":[]}
        try:
            with open(fname,'r') as infile:
                data = json.load(infile)
        except IOError as exc:
            print (str(exc))
            print ("Could not find file. Creating new.")
        data["transition_paths"] += new_paths

        with open(fname,'w') as outfile:
            json.dump(data,outfile)
        self.log( "TSE saved to {}".format(fname) )

    def plot_path_statistics( self, path_file="tse_ensemble.json" ):
        """
        Create a plot to asses convergence of all the paths in the Transition Path Ensemble
        """
        with open(path_file,'r') as infile:
            data = json.load(infile)
        paths = data["transition_paths"]
        total_product_indicator = np.zeros(len(paths[0]["symbols"]))
        total_reactant_indicator = np.zeros(len(paths[0]["symbols"]))
        self.nuc_mc.min_size_product = paths[0]["min_size_product"]
        self.nuc_mc.max_size_reactant = paths[0]["max_size_reactant"]
        for path in paths:
            product_indicator = []
            reactant_indicator = []
            for state in path["symbols"]:
                self.nuc_mc.network.reset()
                self.nuc_mc.set_state( state )
                self.nuc_mc.network(None)
                if ( self.nuc_mc.is_reactant() ):
                    reactant_indicator.append(1)
                else:
                    reactant_indicator.append(0)

                if ( self.nuc_mc.is_product() ):
                    product_indicator.append(1)
                else:
                    product_indicator.append(0)
            total_product_indicator += np.cumsum(product_indicator)/float( len(product_indicator) )
            total_reactant_indicator += np.cumsum(reactant_indicator)/float( len(reactant_indicator) )

        total_reactant_indicator /= float( len(paths) )
        total_product_indicator /= float( len(paths) )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( total_product_indicator, label="Product" )
        ax.plot( total_reactant_indicator, label="Reactant" )
        ax.set_xlabel( "MC sweeps" )
        ax.set_ylabel( "Indicator" )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend( frameon=False, loc="best" )
        return fig
