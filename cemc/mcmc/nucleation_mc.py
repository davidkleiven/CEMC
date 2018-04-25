from cemc.mcmc import Montecarlo
from cemc.mcmc import SGCMonteCarlo
from cemc.mcmc import NetworkObserver
import h5py as h5
import numpy as np
from ase.visualize import view
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
        chem_pot = kwargs.pop("chemical_potential")

        super(NucleationMC,self).__init__(atoms,temp,**kwargs)
        self.chemical_potential = chem_pot
        self.network = NetworkObserver( calc=self.atoms._calc, cluster_name=self.network_name, element=self.network_element )
        self.attach( self.network )

        self.n_bins = self.size_window_width
        self.n_windows = int(self.max_cluster_size/self.size_window_width)
        self.histograms = [np.ones(self.n_bins+1) for _ in range(self.n_windows)]
        self.current_window = 0
        self.mode = Mode.bring_system_into_window

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

    def save( self, fname="nucleation_track.h5" ):
        """
        Saves data to the file
        """
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
        overall_hist = self.merge_histogram()

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

    def merge_histogram(self):
        """
        Merge the histograms
        """
        overall_hist = self.histograms[0].tolist()
        for i in range(1,len(self.histograms)):
            ratio = float(overall_hist[-1])/float(self.histograms[i][0])
            self.histograms[i] *= ratio
            overall_hist += self.histograms[i][1:].tolist()
        return np.array( overall_hist )

    def run( self, nsteps=10000 ):
        """
        Run samples in each window until a desired precission is found
        """
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
