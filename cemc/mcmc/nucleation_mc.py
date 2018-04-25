from cemc.mcmc import Montecarlo
from cemc.mcmc import SGCMonteCarlo
from cemc.mcmc import NetworkObserver
import h5py as h5
import numpy as np
from ase.visualize import view
from scipy.stats import linregress
import os

class Mode(object):
    bring_system_into_window = 0
    sample_in_window = 1
    equillibriate = 2

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
        return move_accepted and self.is_in_window()

    def get_trial_move(self):
        """
        Perform a trial move
        """
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
