from cemc.mcmc import Montecarlo
from cemc.mcmc import SGCMonteCarlo
from cemc.mcmc import NetworkObserver
import h5py as h5

class Mode(object):
    bring_system_into_window = 0
    sample_in_window = 1

class NucleationMC( SGCMonteCarlo ):
    def __init__( self, **kwargs ):
        self.size_window_width = kwargs.pop("size_window_width")
        self.network_name = kwargs.pop("network_name")
        self.network_element = kwargs.pop("network_element")
        self.nbins = kwargs.pop("nbins")
        self.max_cluster_size = kwargs.pop("max_cluster_size")
        super(NucleationMC,self).__init__(**kwargs)
        self.network = NetworkObserver( calc=self.atoms._calc, cluster_name=self.network_name, element=self.network_element )
        self.attach( self.network )

        self.n_windows = int(self.max_cluster_size/self.size_window_width)
        self.histograms = [np.zeros(self.nbins) for _ in range(n_windows)]
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
        return lower,upper

    def is_in_window(self):
        stat = self.network.get_statistics()
        return stat["max_size"] >= self.min_size and stat["max_size"] < self.max_size

    def accept( self, system_changes ):
        accept = Montecarlo.accept( self, system_changes )
        return accept and self.is_in_window()

    def bring_system_into_window(self):
        """
        Brings the system into the current window
        """
        lower,upper = self.get_window_boundaries(self.current_window)
        self.network.grow_cluster( 0.5*(lower+upper) )

    def get_indx( self, size ):
        """
        Get the corresponding bin
        """
        lower,upper = self.get_window_boundaries(self.current_window)
        indx = int( (size-lower)/float(upper-lower) )
        return indx

    def update_histogram(self):
        """
        Update the histogram
        """
        stat = self.network.get_statistics()
        indx = self.get_indx( stat["max_size"] )
        self.histogram[self.current_window][indx] += 1

    def save( self, fname="nucleation_track.h5" ):
        """
        Saves data to the file
        """
        all_data = [np.zeros_like(self.histograms[i]) for _ in range(len(self.histograms))]
        try:
            with h5.File(fname,'r') as hfile:
                for i in range(len(len(self.histograms))):
                    name = "hist{}".format(i)
                    if ( name in hfile ):
                        all_data[i] = hfile[name]
        except Exception as exc:
            print (str(exc))

        for i in range(len(self.histograms)):
            all_data[i] += self.histograms[i]
        self.histograms = all_data
        overall_hist = self.merge_histogram()

        with h5.File(fname,'r+') as hfile:
            for i in range(len(len(self.histograms))):
                name = "hist{}".format(i)
                if ( name in hffile ):
                    data = hf[name]
                    data[...] = all_data[i]
                else:
                    dset = hfile.create_dataset( name, data=all_data[i] )

            if ( "overall_hist" in hfile ):
                data = hf["overall_hist"]
                data[...] = overall_hist
            else:
                dset = hffile.create_dataset( "overall_hist", data=overall_hist )

    def merge_histogram(self):
        """
        Merge the histograms
        """
        overall_hist = self.histograms[0].tolist()
        for i in range(1,len(self.histograms)):
            diff = overall_hist[-1] - self.histpgrams[i][0]
            self.histograms[i] += diff
            overall_hist += self.histograms[i][1:].tolist()
        return np.array( overall_hist )


    def run( self, chem_pot=None, nsteps=10000 ):
        """
        Run samples in each window until a desired precission is found
        """
        for i in range(n_windows):
            self.current_window = 1
            self.reset()
            self.bring_system_into_window()
            self.equillibrate()
            current_step = 0
            while( current_step < nsteps ):
                self._mc_step()
                self.update_histogram()
