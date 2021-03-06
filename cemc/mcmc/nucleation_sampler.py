from cemc.mcmc import NetworkObserver
import h5py as h5
import numpy as np
from ase.visualize import view
from scipy.stats import linregress
import os
from ase.units import kB


class Mode(object):
    bring_system_into_window = 0
    sample_in_window = 1
    equillibriate = 2
    transition_path_sampling = 3


class NucleationSampler(object):
    """
    Class that do the book-keeping needed for free energy calculations of
    nucleation

    :Keyword arguments:
        * *size_window_width* Size range in each window
        * *max_cluster_size* Maximmum cluster size
        * *merge_strategy* How to perform the actual merging (Recommended to
            use the default)
        * *max_one_cluster* Ensure that there is only *one* cluster present in
            the system. For larger cluster sizes this should not matter.
    """
    def __init__(self, **kwargs):
        self.size_window_width = kwargs.pop("size_window_width")
        self.max_cluster_size = kwargs.pop("max_cluster_size")
        self.merge_strategy = "normalize_overlap"
        self.max_one_cluster = False
        self.allow_solutes = True
        if "merge_strategy" in kwargs.keys():
            self.merge_strategy = kwargs.pop("merge_strategy")
        if "max_one_cluster" in kwargs.keys():
            self.max_one_cluster = kwargs.pop("max_one_cluster")

        allowed_merge_strategies = ["normalize_overlap", "fit"]
        if self.merge_strategy not in allowed_merge_strategies:
            msg = "Merge strategy has to be one of {}".format(
                allowed_merge_strategies)
            raise ValueError(msg)
        chem_pot = kwargs.pop("chemical_potential")

        self.n_bins = self.size_window_width
        self.n_windows = int(self.max_cluster_size / self.size_window_width)
        self.histograms = []
        self.singlets = []
        n_singlets = len(chem_pot.keys())
        for i in range(self.n_windows):
            if i == 0:
                self.histograms.append(np.ones(self.n_bins))
                self.singlets.append(np.zeros((self.n_bins, n_singlets)))
            else:
                self.histograms.append(np.ones(self.n_bins + 1))
                self.singlets.append(np.zeros((self.n_bins + 1, n_singlets)))
        self.current_window = 0
        self.mode = Mode.bring_system_into_window
        self.current_cluster_size = 0

    def _get_window_boundaries(self, num):
        """
        Return the upper and lower boundary of the windows

        :param num: Window index
        """
        if num == 0:
            lower = 0
        else:
            lower = (num * self.size_window_width) - 1

        if num == self.n_windows - 1:
            upper = self.max_cluster_size
        else:
            upper = (num + 1) * self.size_window_width
        return int(lower), int(upper)

    def is_in_window(self, network, retstat=False):
        """
        Check if the current network state belongs to the current window

        :param network: Instance of :py:class:`cemc.mcmc.NetworkObserver`
        :param retstat: If true it will also return the network statistics
        """
        network.reset()
        network(None)  # Explicitly call the network observer
        stat = network.get_statistics()
        lower, upper = self._get_window_boundaries(self.current_window)
        max_size_ok = stat["max_size"] >= lower and stat["max_size"] < upper

        n_clusters_ok = True
        if self.max_one_cluster:
            n_clusters = stat["number_of_clusters"]
            n_clusters_ok = (n_clusters == 1)

        if self.mode == Mode.transition_path_sampling:
            max_size_ok = True  # Do not restrict the window size in this case
            n_clusters_ok = True

        network.reset()

        if retstat:
            return max_size_ok and n_clusters_ok, stat
        return max_size_ok and n_clusters_ok

    def bring_system_into_window(self, network):
        """
        Brings the system into the current window

        :param network: Instance of :py:class:`cemc.mcmc.NetworkObserver`
        """
        lower, upper = self._get_window_boundaries(self.current_window)
        size = int(0.5 * (lower + upper) + 1)

        # TODO: Network observers no longer has a grow_cluster function
        # should be handle in python
        network.grow_cluster(size)
        network(None)
        stat = network.get_statistics()
        network.reset()
        if stat["max_size"] != size:
            msg = "Size of created cluster does not match the one requested!\n"
            msg += "Size of created: {}. ".format(stat["max_size"])
            msg += "Requested: {}".format(size)
            raise RuntimeError(msg)
        if stat["number_of_clusters"] != 1:
            msg = "More than one cluster exists!\n"
            msg += "Supposed to create 1 cluster, "
            msg += "created {}".format(stat["number_of_clusters"])
            raise RuntimeError(msg)
        self.current_cluster_size = stat["max_size"]

    def _get_indx(self, size):
        """
        Get the corresponding bin

        :param size: The size of which its corresponding bin number should be
                     computed
        """
        lower, upper = self._get_window_boundaries(self.current_window)
        # indx = int( (size-lower)/float(upper-lower) )
        indx = int(size - lower)
        return indx

    def update_histogram(self, mc_obj):
        """
        Update the histogram

        :param mc_obj: Instance of the sampler
                       (typically `cemc.mcmc.SGCNucleation`)
        """
        stat = mc_obj.network.get_statistics()
        indx = self._get_indx(stat["max_size"])
        if indx < 0:
            lower, upper = self._get_window_boundaries(self.current_window)
            msg = "Given size: {}. ".format(stat["max_size"])
            msg += "Boundaries: [{},{})".format(lower, upper)
            raise ValueError(msg)
        self.histograms[self.current_window][indx] += 1

        if mc_obj.name == "SGCMonteCarlo":
            new_singlets = np.zeros_like(mc_obj.averager.singlets)
            new_singlets = mc_obj.atoms._calc.get_singlets()
            self.singlets[self.current_window][indx, :] += new_singlets

    def helmholtz_free_energy(self, singlets, hist):
        """
        Compute the Helmholtz Free Energy barrier

        :param singlets: Thermal average singlet terms
        :param hist: Histogram of visits
        """
        # N = len(self.atoms)
        # TODO: Fix this
        N = 1000
        beta_gibbs = -np.log(hist)
        beta_helmholtz = beta_gibbs
        beta = 1.0 / (kB * self.T)
        for i in range(len(self.chem_pots)):
            beta_helmholtz += self.chem_pots[i]*singlets[:, i] * N
        beta_helmholtz -= beta_helmholtz[0]
        return beta_helmholtz

    def save(self, fname="nucleation_track.h5"):
        """Save data to the file

        :param fname: Filename should be a HDF5 file
        """

        all_data = [np.zeros_like(self.histograms[i]) for i in
                    range(len(self.histograms))]
        singlets = [np.zeros_like(self.singlets[i]) for i in
                    range(len(self.singlets))]
        try:
            with h5.File(fname, 'r') as hfile:
                for i in range(len(self.histograms)):
                    name = "window{}/hist".format(i)
                    if name in hfile:
                        all_data[i] = np.array(hfile[name])
                    singlet_name = "window{}/singlets".format(i)
                    if name in hfile:
                        singlets[i] = np.array(hfile[singlet_name])
        except Exception as exc:
            print(str(exc))
            print("Creating new file")

        for i in range(len(self.histograms)):
            all_data[i] += self.histograms[i]
            singlets[i] += self.singlets[i]
        self.histograms = all_data
        overall_hist = self.merge_histogram(strategy=self.merge_strategy)
        overall_singlets = self.merge_singlets(singlets, all_data)
        # beta_helm = \
        #    self.helmholtz_free_energy(overall_singlets, overall_hist)
        beta_gibbs = -np.log(overall_hist)
        beta_gibbs -= beta_gibbs[0]

        if os.path.exists(fname):
            flag = "r+"
        else:
            flag = "w"

        with h5.File(fname, flag) as hfile:
            for i in range(len(self.histograms)):
                name = "window{}/hist".format(i)
                if name in hfile:
                    data = hfile[name]
                    data[...] = all_data[i]
                else:
                    dset = hfile.create_dataset(name, data=all_data[i])
                singlet_name = "window{}/singlets".format(i)
                if singlet_name in hfile:
                    data = hfile[singlet_name]
                    data[...] = self.singlets[i]
                else:
                    dset = hfile.create_dataset(singlet_name,
                                                data=self.singlets[i])

            if "overall_hist" in hfile:
                data = hfile["overall_hist"]
                data[...] = overall_hist
            else:
                dset = hfile.create_dataset("overall_hist",
                                            data=overall_hist)

            if "overall_singlets" in hfile:
                data = hfile["overall_singlets"]
                data[...] = overall_singlets
            else:
                dset = hfile.create_dataset("overall_singlets",
                                            data=overall_singlets)

            # if not "chem_pot" in hfile:
            #     dset = hfile.create_dataset("chemical_potentials",
            #                                 data=self.chem_pots)

            # if "beta_helm" in hfile:
            #     data = hfile["beta_helm"]
            #     data[...] = beta_helm
            # else:
            #     dset = hfile.create_dataset("beta_helm", data=beta_helm)
            if "beta_gibbs" in hfile:
                data = hfile["beta_gibbs"]
                data[...] = beta_gibbs
            else:
                dset = hfile.create_dataset("beta_gibbs", data=beta_gibbs)
        self.log("Data saved to {}".format(fname))

    def merge_histogram(self, strategy="normalize_overlap"):
        """
        Merge the histograms

        :param strategy: Which strategy to use when merging
            * *normalize_overlap* Use the last and the first bin of successive
            windows and normalize them such that they are continuous
            * *fit* Perform a linear fit to the last and the first part of two
            successive windows and make them continuous
        """
        overall_hist = self.histograms[0].tolist()

        if strategy == "normalize_overlap":
            for i in range(1, len(self.histograms)):
                ratio = float(overall_hist[-1]) / float(self.histograms[i][0])
                normalized_hist = self.histograms[i] * ratio
                overall_hist += normalized_hist[1:].tolist()
        elif strategy == "fit":
            for i in range(1, len(self.histograms)):
                x1 = [1, 2, 3, 4]
                slope1, interscept1, rvalue1, pvalue1, stderr1 = \
                    linregress(x1, overall_hist[-4:])
                x2 = [4, 5, 6, 7]
                slope2, interscept2, rvalue2, pvalue2, stderr2 = \
                    linregress(x2, self.histograms[i][:4])
                x_eval = np.array([1, 2, 3, 4, 5, 6, 7])
                y1 = slope1*x_eval + interscept1
                y2 = slope2*x_eval + interscept2
                ratio = np.mean(y1 / y2)
                normalized_hist = self.histograms[i] * ratio
                overall_hist += normalized_hist[1:].tolist()
        return np.array(overall_hist)

    def merge_singlets(self, singlets, histograms):
        """
        Merge all the singlets and normalize by the histogram

        :param singlets: Sampled singlets values
        :param histograms: The histograms of *all* windows
        """
        normalized_singlets = []
        for i in range(len(singlets)):
            norm_singl = np.zeros_like(singlets[i])
            for j in range(singlets[i].shape[0]):
                norm_singl[j, :] = singlets[i][j, :] / histograms[i][j]
            normalized_singlets.append(norm_singl)

        all_singlets = normalized_singlets[0]
        for i in range(1, len(normalized_singlets)):
            all_singlets = np.vstack((all_singlets,
                                      normalized_singlets[i][1:, :]))
        return all_singlets

    def log(self, msg):
        """
        Logging

        :param msg: Message to be logged
        """
        print(msg)
