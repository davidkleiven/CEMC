from __future__ import print_function
from cemc.mcmc import SGCNucleation
import json
import numpy as np
from ase.units import kB
from ase.io.trajectory import TrajectoryWriter
import copy
from matplotlib import pyplot as plt
from ase.calculators.singlepoint import SinglePointCalculator
import time

class TransitionPathRelaxer(object):
    def __init__( self, nuc_mc=None ):
        if ( not isinstance(nuc_mc,SGCNucleation) ):
            raise TypeError( "nuc_mc has to be an instance of SGCNucleation" )
        self.nuc_mc = nuc_mc
        self.nuc_mc.set_mode( "transition_path_sampling" )
        self.init_path = None
        self.initial_peak_energy = None
        self.nsteps_per_sweep = None
        self.rank = 0

    def load_path( self, fname ):
        """
        Loads the initial path
        """
        with open(fname,'r') as infile:
            self.init_path = json.load(infile)
        self.initial_peak_energy = np.max(self.init_path["energy"])
        self.nuc_mc.min_size_product = self.init_path["min_size_product"]
        self.nuc_mc.max_size_reactant = self.init_path["max_size_reactant"]
        self.nsteps_per_sweep = self.init_path["nsteps_per_sweep"]

    def shooting_move( self, timeslice ):
        """
        Performs one shooting move
        """
        new_path = {}
        new_path["symbols"] = []
        new_path["energy"] = []
        self.nuc_mc.set_state( self.init_path["symbols"][timeslice] )
        self.nuc_mc.current_energy = self.init_path["energy"][timeslice]
        direction = "nodir"
        N = len(self.init_path["energy"])
        now = time.time()
        output_every = 15
        for i in range(N):
            direction = "transition_region"
            if time.time()-now > output_every:
                self.log("Sweep {} of maximum {}".format(i,N))
                now = time.time()

            self.nuc_mc.network.reset()
            self.nuc_mc.sweep(nsteps=self.nsteps_per_sweep)
            self.nuc_mc.network(None)
            new_path["energy"].append(self.nuc_mc.current_energy)
            new_path["symbols"].append( [atom.symbol for atom in self.nuc_mc.atoms] )
            if self.nuc_mc.is_product():
                # This is a forward path
                if i == len(self.init_path["energy"])-timeslice-2:
                    direction = "forward"
                    break
                elif i > len(self.init_path["energy"])-timeslice-2:
                    # The path is too long and should be rejected
                    direction = "transition_region"
                    break
            elif self.nuc_mc.is_reactant():
                # This is a backward path
                direction = "backward"
                if i == timeslice-1:
                    break
                elif i > timeslice-1:
                    # The path is too long and should be rejected
                    direction = "transition_region"
                    break

        self.nuc_mc.network.reset()
        self.nuc_mc.network(None) # Construct the network to to check the endpoints

        # Figure out if the system ended up in the correct target
        if ( direction == "forward" ):
            self.init_path["energy"][timeslice+1:] = new_path["energy"]
            self.init_path["symbols"][timeslice+1:] = new_path["symbols"]
            self.log ("New forward path accepted." )
            return True
        elif ( direction == "backward" ):
            self.init_path["energy"][:timeslice] = new_path["energy"][::-1]
            self.init_path["symbols"][:timeslice] = new_path["symbols"][::-1]
            self.log ("New backward path accepted." )
            return True
        else:
            self.log( "New path rejected because it did not reach any of the basins")
        return False

    def log(self, msg):
        """
        Log result to a file
        """
        if self.rank == 0:
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
        min_slice = 0
        max_slice = len(self.init_path["energy"])
        for move in range(n_shooting_moves):
            if move%10 == 0:
                self.center_barrier()
                self.path2trajectory()

            self.nuc_mc.reset()
            direct = direcions[np.random.randint(low=0,high=2)]
            self.log( "Move {} of {}".format(move,n_shooting_moves) )
            timeslice = np.random.randint(low=min_slice,high=max_slice )
            self.log("Starting from timeslice {} of {}".format(timeslice,max_slice))
            self.shooting_move(timeslice)

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
        for energy,state in zip(self.init_path["energy"], self.init_path["symbols"]):
            self.nuc_mc.network.reset()
            self.nuc_mc.set_state(state)
            self.nuc_mc.network(None)
            atoms = self.nuc_mc.network.get_atoms_with_largest_cluster( prohibited_symbols=["Al","Mg"] )
            if atoms is None:
                atoms = self.nuc_mc.atoms
            calc = SinglePointCalculator(atoms, energy=energy)
            traj.write(atoms)
        self.log( "Trajectory written to {}".format(fname))

    def center_barrier( self, verbose=False ):
        """
        Refine the path length such that the time spent in the reactant basin
        is equal to the time spent in the product basin
        """
        reactant_indicator, product_indicator = self.get_basin_indicators(self.init_path)
        n_react = np.sum(reactant_indicator)
        n_prod = np.sum(product_indicator)
        diff = np.abs(n_react-n_prod)
        delta = int(diff/2)
        basin = ""
        if ( n_react > n_prod ):
            # Remove the first slices from the reactant side
            self.init_path["energy"] = self.init_path["energy"][delta:]
            self.init_path["symbols"] = self.init_path["symbols"][delta:]
            self.nuc_mc.set_state( self.init_path["symbols"][-1] )
            self.nuc_mc.current_energy = self.init_path["energy"][-1]
            basin = "product"
        elif ( n_prod > n_react ):
            # Remove the last slices from the product side
            self.init_path["energy"] = self.init_path["energy"][:-delta]
            self.init_path["symbols"] = self.init_path["symbols"][:-delta]
            self.nuc_mc.set_state( self.init_path["symbols"][0] )
            self.nuc_mc.current_energy = self.init_path["energy"][0]
            basin = "reactant"

        new_path = {"symbols":[], "energy":[]}
        for i in range(delta):
            self.nuc_mc.network.reset()
            self.nuc_mc.sweep(nsteps=self.nsteps_per_sweep)
            self.nuc_mc.network(None)
            new_path["energy"].append(self.nuc_mc.current_energy)
            new_path["symbols"].append( [atom.symbol for atom in self.nuc_mc.atoms] )

            if basin == "reactant":
                if not self.nuc_mc.is_reactant():
                    raise RuntimeError("System leaving reactants, when starting inside the basin!")
            elif basin == "product":
                if not self.nuc_mc.is_product():
                    raise RuntimeError("System leaving products when starting inside basin!")

        if basin == "reactant":
            self.log("Inserting {} states in the beginning of the trajectory".format(delta))
            self.init_path["energy"] = new_path["energy"][::-1]+self.init_path["energy"]
            self.init_path["symbols"] = new_path["symbols"][::-1]+self.init_path["symbols"]
        else:
            self.init_path["energy"] = self.init_path["energy"]+new_path["energy"]
            self.init_path["symbols"] = self.init_path["symbols"]+new_path["symbols"]
            self.log("Appending {} states to the end of the trajectory".format(delta))


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

        if min_slice_outside_reactant is None:
            min_slice_outside_reactant = 0
        if max_slice_outside_products is None:
            max_slice_outside_products = len(self.init_path["energy"])
        return min_slice_outside_reactant,max_slice_outside_products


    def generate_paths( self, initial_path=None, n_paths=1, max_attempts=10000, outfile="tse_ensemble.json", mpicomm=None ):
        """
        Generate a given number of paths
        """
        all_paths = []
        self.load_path(initial_path)
        orig_path = copy.deepcopy(self.init_path)
        if ( n_paths >= max_attempts ):
            raise ValueError( "The number of paths requested exceeds the maximum number of attempts. Increase the maximum number of attempts" )
        counter = 0
        min_slice,max_slice = self.find_timeslices_in_transition_region()

        n_paths_found = 0
        overall_num_paths = 0
        if mpicomm is not None:
            self.rank = mpicomm.Get_rank()
        while( overall_num_paths < n_paths and counter < max_attempts ):
            self.log("Total number of paths found: {}".format(overall_num_paths))
            counter += 1
            self.init_path = copy.deepcopy(orig_path)
            timeslice = np.random.randint(low=min_slice,high=max_slice)
            if ( self.shooting_move(timeslice) ):
                all_paths.append(self.init_path)
                n_paths_found += 1

            if mpicomm is not None:
                send_buf = np.zeros(1)
                send_buf[0] = n_paths_found
                recv_buf = np.zeros(1)
                mpicomm.Allreduce(send_buf, recv_buf)
                overall_num_paths = recv_buf[0]
            else:
                overall_num_paths = n_paths_found

        if mpicomm is not None:
            temp_paths = mpicomm.gather(all_paths, root=0)
            all_paths = []
            if self.rank == 0:
                for sublist in temp_paths:
                    all_paths += sublist

        if self.rank == 0:
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

    def get_basin_indicators(self,path):
        """
        Compute the basin indicators of the state
        """
        reactant_indicator = []
        product_indicator = []
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
        return reactant_indicator, product_indicator

    def plot_path_statistics( self, path_file="tse_ensemble.json" ):
        """
        Create a plot to asses convergence of all the paths in the Transition Path Ensemble
        """
        with open(path_file,'r') as infile:
            data = json.load(infile)
        try:
            paths = data["transition_paths"]
        except KeyError:
            paths = [data]
        total_product_indicator = np.zeros(len(paths[0]["symbols"]))
        total_reactant_indicator = np.zeros(len(paths[0]["symbols"]))
        self.nuc_mc.min_size_product = paths[0]["min_size_product"]
        self.nuc_mc.max_size_reactant = paths[0]["max_size_reactant"]
        for path in paths:
            product_indicator = []
            reactant_indicator = []
            reactant_indicator, product_indicator = self.get_basin_indicators(path)

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
