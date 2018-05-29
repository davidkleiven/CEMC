from cemc.mcmc import NetworkObserver
import h5py as h5
import numpy as np
from ase.visualize import view
from scipy.stats import linregress
import os
from ase.units import kB
import mpi_tools

class Mode(object):
    bring_system_into_window = 0
    sample_in_window = 1
    equillibriate = 2

class NucleationSampler( object ):
    def __init__( self, **kwargs ):
        self.size_window_width = kwargs.pop("size_window_width")
        self.max_cluster_size = kwargs.pop("max_cluster_size")
        self.merge_strategy = "normalize_overlap"
        self.max_one_cluster = False
        self.allow_solutes = True
        if ( "merge_strategy" in kwargs.keys() ):
            self.merge_strategy = kwargs.pop("merge_strategy")
        if ( "max_one_cluster" in kwargs.keys() ):
            self.max_one_cluster = kwargs.pop("max_one_cluster")

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

        self.n_bins = self.size_window_width
        self.n_windows = int(self.max_cluster_size/self.size_window_width)
        self.histograms = []
        self.singlets = []
        n_singlets = len(chem_pot.keys())
        for i in range(self.n_windows):
            if ( i==0 ):
                self.histograms.append( np.ones(self.n_bins) )
                self.singlets.append( np.zeros((self.n_bins,n_singlets)) )
            else:
                self.histograms.append( np.ones(self.n_bins+1) )
                self.singlets.append( np.zeros((self.n_bins+1,n_singlets)) )
        self.current_window = 0
        self.mode = Mode.bring_system_into_window
        mpi_tools.set_seeds( self.nucleation_mpicomm )
        self.current_cluster_size = 0

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

    def is_in_window(self,network,retstat=False):
        network.reset()
        network(None) # Explicitly call the network observer
        stat = network.get_statistics()
        lower,upper = self.get_window_boundaries(self.current_window)
        max_size_ok = stat["max_size"] >= lower and stat["max_size"] < upper
        network.reset()
        if ( self.max_one_cluster ):
            n_clusters = stat["number_of_clusters"]
            if ( retstat ):
                return max_size_ok and n_clusters == 1,stat
            else:
                return max_size_ok and n_clusters == 1
        if ( retstat ):
            return max_size_ok,stat
        return max_size_ok

    def bring_system_into_window(self,network):
        """
        Brings the system into the current window
        """
        lower,upper = self.get_window_boundaries(self.current_window)
        size = int(0.5*(lower+upper)+1)
        network.grow_cluster( size )
        network(None)
        stat = network.get_statistics()
        network.reset()
        if ( stat["max_size"] != size ):
            msg = "The size of the cluster created does not match the one requested!\n"
            msg += "Size of created: {}. Requested: {}".format(stat["max_size"],size)
            raise RuntimeError( msg )
        if ( stat["number_of_clusters"] != 1 ):
            msg = "More than one cluster exists!\n"
            msg += "Was supposed to create 1 cluster, created {}".format(stat["number_of_clusters"])
            raise RuntimeError(msg)
        self.current_cluster_size = stat["max_size"]

    def get_indx( self, size ):
        """
        Get the corresponding bin
        """
        lower,upper = self.get_window_boundaries(self.current_window)
        #indx = int( (size-lower)/float(upper-lower) )
        indx = int(size-lower)
        return indx

    def update_histogram(self,mc_obj):
        """
        Update the histogram
        """
        stat = mc_obj.network.get_statistics()
        indx = self.get_indx( stat["max_size"] )
        if ( indx < 0 ):
            lower,upper = self.get_window_boundaries(self.current_window)
            raise ValueError( "Given size: {}. Boundaries: [{},{})".format(stat["max_size"],lower,upper))
        self.histograms[self.current_window][indx] += 1

        if ( mc_obj.name == "SGCMonteCarlo" ):
            new_singlets = np.zeros_like( mc_obj.averager.singlets )
            mc_obj.atoms._calc.get_singlets(  new_singlets )
            self.singlets[self.current_window][indx,:] += new_singlets

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

            recv_buf = np.zeros_like(self.singlets[i])
            send_buf = self.singlets[i]
            self.nucleation_mpicomm.Allreduce( send_buf, recv_buf )
            self.singlets[i][:,:] = recv_buf[:,:]

    def helmholtz_free_energy(self,singlets,hist):
        """
        Compute the Helmholtz Free Energy barrier
        """
        #N = len(self.atoms)
        # TODO: Fix this
        N = 1000
        beta_gibbs = -np.log(hist)
        beta_helmholtz = beta_gibbs
        beta = 1.0/(kB*self.T)
        for i in range(len(self.chem_pots)):
            beta_helmholtz += self.chem_pots[i]*singlets[:,i]*N
        beta_helmholtz -= beta_helmholtz[0]
        return beta_helmholtz

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
            singlets = [np.zeros_like(self.singlets[i]) for i in range(len(self.singlets))]
            try:
                with h5.File(fname,'r') as hfile:
                    for i in range(len(self.histograms)):
                        name = "window{}/hist".format(i)
                        if ( name in hfile ):
                            all_data[i] = np.array( hfile[name] )
                        singlet_name = "window{}/singlets".format(i)
                        if ( name in hfile ):
                            singlets[i] = np.array(hfile[singlet_name])
            except Exception as exc:
                print (str(exc))
                print ("Creating new file")

            for i in range(len(self.histograms)):
                all_data[i] += self.histograms[i]
                singlets[i] += self.singlets[i]
            self.histograms = all_data
            overall_hist = self.merge_histogram(strategy=self.merge_strategy)
            overall_singlets = self.merge_singlets( singlets, all_data )
            #beta_helm = self.helmholtz_free_energy( overall_singlets, overall_hist )
            beta_gibbs = -np.log(overall_hist)
            beta_gibbs -= beta_gibbs[0]

            if ( os.path.exists(fname) ):
                flag = "r+"
            else:
                flag = "w"

            with h5.File(fname,flag) as hfile:
                for i in range(len(self.histograms)):
                    name = "window{}/hist".format(i)
                    if ( name in hfile ):
                        data = hfile[name]
                        data[...] = all_data[i]
                    else:
                        dset = hfile.create_dataset( name, data=all_data[i] )
                    singlet_name = "window{}/singlets".format(i)
                    if ( singlet_name in hfile ):
                        data = hfile[singlet_name]
                        data[...] = self.singlets[i]
                    else:
                        dset = hfile.create_dataset( singlet_name, data=self.singlets[i] )

                if ( "overall_hist" in hfile ):
                    data = hfile["overall_hist"]
                    data[...] = overall_hist
                else:
                    dset = hfile.create_dataset( "overall_hist", data=overall_hist )

                if ( "overall_singlets" in hfile ):
                    data = hfile["overall_singlets"]
                    data[...] = overall_singlets
                else:
                    dset = hfile.create_dataset( "overall_singlets", data=overall_singlets )

                #if ( not "chem_pot" in hfile ):
                #    dset = hfile.create_dataset( "chemical_potentials", data=self.chem_pots )

                #if ( "beta_helm" in hfile ):
                #    data = hfile["beta_helm"]
                #    data[...] = beta_helm
                #else:
                #    dset = hfile.create_dataset( "beta_helm", data=beta_helm )
                if ( "beta_gibbs" in hfile ):
                    data = hfile["beta_gibbs"]
                    data[...] = beta_gibbs
                else:
                    dset = hfile.create_dataset( "beta_gibbs", data=beta_gibbs )
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

    def merge_singlets( self, singlets, histograms ):
        """
        Merge all the singlets and normalize by the histogram
        """
        normalized_singlets = []
        for i in range(len(singlets)):
            norm_singl = np.zeros_like(singlets[i])
            for j in range(singlets[i].shape[0]):
                norm_singl[j,:] = singlets[i][j,:]/histograms[i][j]
            normalized_singlets.append(norm_singl)

        all_singlets = normalized_singlets[0]
        for i in range(1,len(normalized_singlets)):
            all_singlets = np.vstack((all_singlets,normalized_singlets[i][1:,:]))
        return all_singlets

    def log(self,msg):
        """
        Logging
        """
        print(msg)
