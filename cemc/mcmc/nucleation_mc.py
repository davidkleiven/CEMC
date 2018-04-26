from cemc.mcmc import Montecarlo
from cemc.mcmc import SGCMonteCarlo
from cemc.mcmc import NetworkObserver
import h5py as h5
import numpy as np
from ase.visualize import view
from scipy.stats import linregress
import os
from ase.io.trajectory import TrajectoryWriter
import time
import json
import copy

class Mode(object):
    bring_system_into_window = 0
    sample_in_window = 1
    equillibriate = 2
    transition_path_sampling = 3

class DidNotReachProductOrReactantError(Exception):
    def __init__(self,msg):
        super(DidNotReachProductOrReactantError,self).__init__(msg)

class NucleationMC( SGCMonteCarlo ):
    def __init__( self, atoms, temp, **kwargs ):
        self.size_window_width = kwargs.pop("size_window_width")
        self.network_name = kwargs.pop("network_name")
        self.network_element = kwargs.pop("network_element")
        self.max_cluster_size = kwargs.pop("max_cluster_size")
        self.merge_strategy = "normalize_overlap"
        if ( "merge_strategy" in kwargs.keys() ):
            self.merge_strategy = kwargs.pop("merge_strategy")

        # The Nucleation barrier algorithm requires a lot of communication if
        # parallelized in the same way as SGCMonteCarlo
        # therefore we snap the mpicomm object passed,
        # and parallelize in a different way
        self.nucleation_mpicomm = None
        if ( "mpicomm" in kwargs.keys() ):
            self.nucleation_mpicomm = kwargs.pop("mpicomm")

        allowed_merge_strategies = ["normalize_overlap","fit"]
        if ( self.merge_strategy not in allowed_merge_strategies ):
            raise ValueError( "Merge strategy has to be one of {}".format(allowed_merge_strategies))
        chem_pot = kwargs.pop("chemical_potential")

        super(NucleationMC,self).__init__(atoms,temp,**kwargs)
        self.chemical_potential = chem_pot
        self.network = NetworkObserver( calc=self.atoms._calc, cluster_name=self.network_name, element=self.network_element )
        self.attach( self.network )

        self.n_bins = self.size_window_width
        self.n_windows = int(self.max_cluster_size/self.size_window_width)
        self.histograms = []
        for i in range(self.n_windows):
            if ( i==0 ):
                self.histograms.append( np.ones(self.n_bins) )
            else:
                self.histograms.append( np.ones(self.n_bins+1) )
        self.current_window = 0
        self.mode = Mode.bring_system_into_window
        self.set_seeds( self.nucleation_mpicomm )

        # Variables used for transition state sampling
        self.max_size_reactant = None
        self.min_size_product = None

    def get_window_boundaries(self, num):
        """
        Return the upper and lower boundary of the windows
        """
        if ( num == 0 ):
            lower = 0
        else:
            lower = num*self.size_window_width-1

        if ( num == self.n_windows-1 ):
            upper = self.max_cluster_size
        else:
            upper = (num+1)*self.size_window_width
        return int(lower),int(upper)

    def is_in_window(self):
        self.network(None) # Explicitly call the network observer
        stat = self.network.get_statistics()
        lower,upper = self.get_window_boundaries(self.current_window)

        # During equillibiriation we have to reset the network statistics here
        #if ( self.mode == Mode.equillibriate ):
        self.network.reset()
        #print ("{} <= {} < {}".format(lower,stat["max_size"],upper))
        return stat["max_size"] >= lower and stat["max_size"] < upper

    def accept( self, system_changes ):
        move_accepted = Montecarlo.accept( self, system_changes )
        if ( self.mode == Mode.transition_path_sampling ):
            return move_accepted
        return move_accepted and self.is_in_window()

    def get_trial_move(self):
        """
        Perform a trial move
        """
        if ( self.mode != Mode.transition_path_sampling ):
            if ( not self.is_in_window() ):
                raise RuntimeError( "System is outside the window before the trial move is performed!" )
        return SGCMonteCarlo.get_trial_move(self)

    def bring_system_into_window(self):
        """
        Brings the system into the current window
        """
        lower,upper = self.get_window_boundaries(self.current_window)
        self.network.grow_cluster( int(0.5*(lower+upper)) )

    def get_indx( self, size ):
        """
        Get the corresponding bin
        """
        lower,upper = self.get_window_boundaries(self.current_window)
        #indx = int( (size-lower)/float(upper-lower) )
        indx = int(size-lower)
        return indx

    def update_histogram(self):
        """
        Update the histogram
        """
        stat = self.network.get_statistics()
        indx = self.get_indx( stat["max_size"] )
        if ( indx < 0 ):
            lower,upper = self.get_window_boundaries(self.current_window)
            raise ValueError( "Given size: {}. Boundaries: [{},{})".format(stat["max_size"],lower,upper))
        self.histograms[self.current_window][indx] += 1

    def collect_results(self):
        """
        Collect results from all processors if this algorithm is run in parallel
        """
        if ( self.nucleation_mpicomm is None ):
            return

        for i in range(len(self.histograms)):
            recv_buf = np.zeros_like(self.histograms[i])
            send_buf = self.histograms[i]
            self.nucleation_mpicomm.Allreduce( send_buf, recv_buf )
            self.histograms[i][:] = recv_buf[:]

    def save( self, fname="nucleation_track.h5" ):
        """
        Saves data to the file
        """
        self.collect_results()
        rank = 0
        if ( self.nucleation_mpicomm is not None ):
            rank = self.nucleation_mpicomm.Get_rank()

        if ( rank == 0 ):
            all_data = [np.zeros_like(self.histograms[i]) for i in range(len(self.histograms))]
            try:
                with h5.File(fname,'r') as hfile:
                    for i in range(len(self.histograms)):
                        name = "hist{}".format(i)
                        if ( name in hfile ):
                            all_data[i] = np.array( hfile[name] )
            except Exception as exc:
                print (str(exc))
                print ("Creating new file")

            for i in range(len(self.histograms)):
                all_data[i] += self.histograms[i]
            self.histograms = all_data
            overall_hist = self.merge_histogram(strategy=self.merge_strategy)

            if ( os.path.exists(fname) ):
                flag = "r+"
            else:
                flag = "w"

            with h5.File(fname,flag) as hfile:
                for i in range(len(self.histograms)):
                    name = "hist{}".format(i)
                    if ( name in hfile ):
                        data = hfile[name]
                        data[...] = all_data[i]
                    else:
                        dset = hfile.create_dataset( name, data=all_data[i] )

                if ( "overall_hist" in hfile ):
                    data = hfile["overall_hist"]
                    data[...] = overall_hist
                else:
                    dset = hfile.create_dataset( "overall_hist", data=overall_hist )
            self.log( "Data saved to {}".format(fname) )

    def merge_histogram(self,strategy="normalize_overlap"):
        """
        Merge the histograms
        """
        overall_hist = self.histograms[0].tolist()

        if ( strategy == "normalize_overlap" ):
            for i in range(1,len(self.histograms)):
                ratio = float(overall_hist[-1])/float(self.histograms[i][0])
                normalized_hist = self.histograms[i]*ratio
                overall_hist += normalized_hist[1:].tolist()
        elif ( strategy == "fit" ):
            for i in range(1,len(self.histograms)):
                x1 = [1,2,3,4]
                slope1,interscept1,rvalue1,pvalue1,stderr1 = linregress(x1,overall_hist[-4:])
                x2 = [4,5,6,7]
                slope2,interscept2,rvalue2,pvalue2,stderr2 = linregress(x2,self.histograms[i][:4])
                x_eval = np.array([1,2,3,4,5,6,7])
                y1 = slope1*x_eval + interscept1
                y2 = slope2*x_eval + interscept2
                ratio = np.mean( y1/y2 )
                normalized_hist = self.histograms[i]*ratio
                overall_hist += normalized_hist[1:].tolist()
        return np.array( overall_hist )

    def run( self, nsteps=10000 ):
        """
        Run samples in each window until a desired precission is found
        """
        self.remove_network_observers()
        self.attach(self.network)

        if ( self.nucleation_mpicomm is not None ):
            self.nucleation_mpicomm.barrier()
        for i in range(self.n_windows):
            self.log( "Window {} of {}".format(i,self.n_windows) )
            self.current_window = i
            self.reset()
            self.bring_system_into_window()

            self.mode = Mode.equillibriate
            self.estimate_correlation_time()
            self.equillibriate()
            self.mode = Mode.sample_in_window

            current_step = 0
            while( current_step < nsteps ):
                current_step += 1
                self._mc_step()
                self.update_histogram()
                self.network.reset()

        if ( self.nucleation_mpicomm is not None ):
            self.nucleation_mpicomm.barrier()

    def remove_snapshot_observers(self):
        """
        Remove all Snapshot observers from the observers
        """
        self.observers = [obs for obs in self.observers if obs.name != "Snapshot"]

    def remove_network_observers(self):
        """
        Remove NetworkObservers
        """
        self.observers = [obs for obs in self.observers if obs[1].name != "NetworkObserver"]

    def is_reactant(self):
        """
        Returns true if the current state is in the reactant region
        """
        if ( self.max_size_reactant is None ):
            raise ValueError( "Maximum cluster size to be characterized as reactant is not set!" )

        stat = self.network.get_statistics()
        return stat["max_size"] < self.max_size_reactant

    def is_product(self):
        """
        Return True if the current state is a product state
        """
        if ( self.min_size_product is None ):
            raise ValueError( "Minimum cluster size to be characterized as product is not set!" )
        stat = self.network.get_statistics()
        return stat["max_size"] >= self.min_size_product

    def merge_product_and_reactant_path( self, reactant_traj, product_traj, reactant_symb, product_symb ):
        """
        Merge the product and reactant path into one file
        """
        folder = reactant_traj.rpartition("/")[0]
        symb_merged = folder+"/reaction2product.txt"
        reactant_symbols = []
        product_symbols = []

    def save_list_of_lists(self,fname,data):
        """
        Save a list of lists into a text file
        """
        with open(fname,'w') as outfile:
            for sublist in data:
                for entry in sublist:
                    outfile.write("{} ".format(entry))
                outfile.write("\n")

    def read_list_of_lists(self,fname,dtype="str"):
        """
        Read list of lists
        """
        supported_dtypes = ["str"]
        if ( dtype not in supported_dtypes ):
            raise ValueError( "dtype hsa to be one of {}".format(supported_dtypes))

    def symbols2uint( self, symbols, description ):
        """
        Convert an array of symbols into a numpy array of indices to the desctiption array
        """
        nparray = np.zeros( len(symbols), dtype=np.uint8 )
        for i,symb in enumerate(symbols):
            nparray[i] = desctiption.index(symb)
        return nparray

    def uint2symbols( self, nparray, description ):
        """
        Convert uint8 array to symbols array
        """
        symbs = []
        for i in range(len(nparray)):
            symbs.append( description[nparray[i]] )
        return symbs

    def merge_reference_path( self, res_reactant, res_product ):
        """
        This store the reference path into a JSON
        """
        res_reactant["energy"] = res_reactant["energy"][::-1]
        res_reactant["symbols"] = res_reactant["symbols"][::-1]
        combined_path = {}
        combined_path["energy"] = res_reactant["energy"]+res_product["energy"]
        combined_path["symbols"] = res_reactant["symbols"]+res_product["symbols"]
        return combined_path

    def save_path( self, fname, res ):
        """
        Stores the path result to a JSON file
        """
        with open(fname,'w') as outfile:
            json.dump(res,outfile)


    def find_transition_path( self, initial_cluster_size=None, max_size_reactant=None, min_size_product=None, path_length=1000, max_attempts=100, folder="." ):
        """
        Find one transition path
        """
        if ( initial_cluster_size is None ):
            raise ValueError( "Initial cluster size not given!" )
        if ( max_size_reactant is None ):
            raise ValueError( "The maximum cluster size allowed for the state to be characterized as reactant is not given!" )
        if ( min_size_product is None ):
            raise ValueError( "The minimum size of cluster allowed for the state to be characterized as product is not given!" )

        self.mode = Mode.transition_path_sampling
        self.max_size_reactant = max_size_reactant
        self.min_size_product = min_size_product

        found_reactant_origin = False
        found_product_origin = False
        self.remove_network_observers()
        self.attach( self.network, interval=len(self.atoms) )

        num_reactants = 0
        num_products = 0
        default_trajfile = folder+"/default_trajfile.traj"
        reactant_file = folder+"/trajectory_reactant.traj"
        product_file = folder+"/trajectory_product.traj"
        reference_path_file = folder+"/reference_path.json"

        self.network.reset()
        self.network.grow_cluster( initial_cluster_size )

        init_symbols = [atom.symbol for atom in self.atoms]
        target = "both"
        reactant_res = {}
        product_res = {}
        for attempt in range(max_attempts):
            self.reset()
            self.atoms._calc.set_symbols(init_symbols)
            try:
                res = self.find_one_transition_path( path_length=path_length, trajfile=default_trajfile, target=target )
            except DidNotReachProductOrReactantError as exc:
                self.log( str(exc) )
                self.log ( "Trying one more time" )
                continue

            if ( res["type"] == "reactant" ):
                num_reactants += 1
                target = "product" # Reactant is found, search only for products
                if ( not found_reactant_origin ):
                    os.rename( default_trajfile,reactant_file)
                    found_reactant_origin = True
                    reactant_res = copy.deepcopy(res)

            elif ( res["type"] == "product" ):
                num_products += 1
                target = "reactant" # Product is found search only for reactant
                if ( not found_product_origin ):
                    os.rename( default_trajfile,product_file)
                    found_product_origin = True
                    product_res = copy.deepcopy(res)

            if ( os.path.exists(default_trajfile) ):
                os.remove(default_trajfile)

            if ( found_product_origin and found_reactant_origin ):
                combined_path = self.merge_reference_path(reactant_res,product_res)
                self.save_path( reference_path_file, combined_path )
                self.log( "Found a path to the product region and a path to the reactant region" )
                self.log( "They are stored in {} and {}".format(product_file,reactant_file))
                self.log( "The reference path is stored in {}".format(reference_path_file) )
                return
            self.log( "Attempt: {} of {} ended in {} region".format(attempt,max_attempts,res["type"]) )
        msg = "Did not manage to find both a configuration in the product region and the reactant region\n"
        raise RuntimeError( msg )


    def find_one_transition_path( self, path_length=1000, trajfile="default.traj", target="both" ):
        """
        Finds a transition path by running random samples
        """
        supported_targets = ["reactant","product","both"]
        if ( target not in supported_targets ):
            raise ValueError( "Target has to be one of {}".format(supported_targets) )

        # Check if a snapshot tracker is attached
        traj = TrajectoryWriter( trajfile, mode="w" )
        current_step = 0
        result = {}
        symbs = []
        unique_symbols = []
        for atom in self.atoms:
            if ( atom.symbol not in unique_symbols ):
                unique_symbols.append(atom.symbol)

        output_every_sec = 30
        now = time.time()
        energies = []
        result = {}
        for sweep in range(path_length):
            self.network.reset()
            if ( time.time() - now > output_every_sec ):
                self.log( "Sweep {} of {}".format(sweep,path_length))
                now = time.time()
            for step in range(len(self.atoms)):
                self._mc_step()
            self.network(None) # Explicitly enforce a construction of the network
            energies.append(self.current_energy)
            symbs.append( [atom.symbol for atom in self.atoms] )
            atoms = self.network.get_atoms_with_largest_cluster(prohibited_symbols=unique_symbols)
            if ( atoms is None ):
                traj.write(self.atoms)
            else:
                traj.write(atoms)

            if ( target == "reactant" ):
                if ( self.is_product() ):
                    # Terminate before the desired path length is reached
                    result["type"] = "reactant"
                    result["symbols"] = symbs
                    result["energy"] = energies
                    return result
            elif ( target == "product" ):
                if ( self.is_reactant() ):
                    result["type"] = "product"
                    result["symbols"] = symbs
                    result["energy"] = energies
                    # Terminate before the desired path length is reached
                    return result

        traj.close()
        if ( self.is_reactant() ):
            result["type"] = "reactant"
        elif( self.is_product() ):
            result["type"] = "product"
        else:
            stat = self.network.get_statistics()
            max_size = stat["max_size"]
            msg = "State did not end up in product or reactant region. Increase the number of sweeps.\n"
            msg += "Max. cluster size {}. Max cluster size reactants {}. Min cluster size products {}".format(max_size,self.max_size_reactant,self.min_size_product)
            raise DidNotReachProductOrReactantError( msg )

        result["symbols"] = symbs
        result["energy"] = energies
        return result
