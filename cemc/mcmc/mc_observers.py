import copy
import numpy as np
from cemc_cpp_code import PyClusterTracker
from ase.io.trajectory import TrajectoryWriter
from cemc.mcmc.averager import Averager
from cemc.mcmc.util import waste_recycled_average
from cemc.mcmc.mpi_tools import num_processors, mpi_rank
highlight_elements = ["Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
                      "Al", "Si", "P", "S", "Cl", "Ar"]


class MCObserver(object):
    """Base class for all MC observers."""

    def __init__(self):
        self.name = "GenericObserver"

    def __call__(self, system_changes):
        """
        Gets information about the system changes and can perform some action

        :param list system_changes: List of system changes if indx 23 changed
            from Mg to Al this argument would be
            [(23, Mg, Al)]
            If site 26 with an Mg atom is swapped with site 12 with an Al atom
            this would be
            [(26, Mg, Al), (12, Al, Mg)]
        """
        pass

    def reset(self):
        """Reset all values of the MC observer"""
        pass


class CorrelationFunctionTracker(MCObserver):
    """
    Track the history of the correlation function.
    Only relevant if the calculator is a CE calculator

    :param CE ce_calc: Instance of the CE calculator attached to the atoms
        object
    """

    def __init__(self, ce_calc):
        self.cf = []
        self.ce_calc = ce_calc
        self.name = "CorrelationFunctionTracker"

    def __call__(self, system_changes):
        """Update the correlation functions."""
        self.cf.append(copy.deepcopy(self.ce_calc.cf))

    def plot_history(self, max_size=10):
        """Plot history (only if history is tracked).

        :param int max_size: Maximum cluster size to include

        :return: Figure with the plot
        :rtype: Figure
        """
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for key in self.cf[0].keys():
            size = int(key[1])
            if size > max_size:
                continue
            cf_history = [cf[key] for cf in self.cf]
            ax.plot(cf_history, label=key, ls="steps")
        ax.legend(loc="best", frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return fig


class PairCorrelationObserver(MCObserver):
    """Compute the average value of all the ECIs.

    :param CE ce_calc: CE calculator object
    """

    def __init__(self, ce_calc):
        self.cf = {}
        self.cf_squared = {}
        self.ce_calc = ce_calc
        if self.ce_calc.updater is None:
            msg = "This observer can only be used with the C++ version of the "
            msg += "CF updater"
            raise RuntimeError(msg)
        self.n_entries = 0
        self.name = "PairCorrelationObserver"

        for key in self.ce_calc.eci.keys():
            if key.startswith("c2_"):
                self.cf[key] = 0.0
                self.cf_squared[key] = 0.0

    def __call__(self, system_changes):
        """Update correlation functions.

        :param list system_changes: Last changes to the system
        """
        new_cf = self.ce_calc.updater.get_cf()
        self.n_entries += 1
        for key in self.cf.keys():
            self.cf[key] += new_cf[key]
            self.cf_squared[key] += new_cf[key]**2

    def get_average(self):
        """Returns the average.

        :return: Thermal averaged correlation functions
        :rtype: dict
        """
        avg_cf = copy.deepcopy(self.cf)
        for key in avg_cf.keys():
            avg_cf[key] /= self.n_entries
        return avg_cf

    def get_std(self):
        """Return the standard deviation.

        :return: Standard deviation of the correlation functions
        :rtype: dict
        """
        std_cf = {key: 0.0 for key in self.cf.keys()}
        for key in self.cf.keys():
            std_cf[key] = np.sqrt(self.cf_squared[key] / self.n_entries
                                  - (self.cf[key] / self.n_entries)**2)
        return std_cf


class LowestEnergyStructure(MCObserver):
    """
    Observer that tracks the lowest energy state visited
    during an MC run

    :param CE ce_calc: Instance of the CE calculator
    :param Montecarlo mc_obj: Monte Carlo object
    """

    def __init__(self, ce_calc, mc_obj, verbose=False):
        self.ce_calc = ce_calc
        self.mc_obj = mc_obj
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.atoms = None
        # Always the same as atoms. Included for backward compatibility
        self.lowest_energy_atoms = None
        self.name = "LowestEnergyStructure"
        self.verbose = verbose

    def __call__(self, system_changes):
        """
        Checks if the current state has lower energy.
        If it has lower energy, the new state will be stored

        :param list system_changes: Last changes to the system
        """
        if (self.atoms is None or self.lowest_energy_cf is None):
            self.lowest_energy_cf = self.ce_calc.get_cf()
            self.lowest_energy = self.mc_obj.current_energy
            self.atoms = self.mc_obj.atoms.copy()
            self.lowest_energy_atoms = self.atoms
            return

        if self.mc_obj.current_energy < self.lowest_energy:
            dE = self.mc_obj.current_energy - self.lowest_energy
            self.lowest_energy = self.mc_obj.current_energy
            self.atoms = self.mc_obj.atoms.copy()
            self.lowest_energy_atoms = self.atoms  # For backward compatibility
            self.lowest_energy_cf = self.ce_calc.get_cf()
            if self.verbose:
                msg = "Found new low energy structure. "
                msg += "New energy: {} eV. ".format(self.lowest_energy)
                msg += "Change: {} eV".format(dE)
                print(msg)


class SGCObserver(MCObserver):
    """
    Observer mainly intended to track additional quantities needed when
    running SGC Monte Carlo

    :param CE ce_calc: CE calculator
    :param SGCMonteCarlo mc_obj: Instance of the Monte Carlo object
    :param int n_singlets: Number of singlet terms to track
    """

    def __init__(self, ce_calc, mc_obj, n_singlets):
        super(SGCObserver, self).__init__()
        self.name = "SGCObersver"
        self.ce_calc = ce_calc
        self.mc = mc_obj
        self.recycle_waste = self.mc.recycle_waste

        self.quantities = {
            "singlets": np.zeros(n_singlets, dtype=np.float64),
            "singlets_sq": np.zeros(n_singlets, dtype=np.float64),
            "energy": Averager(ref_value=ce_calc.get_energy()),
            "energy_sq": Averager(ref_value=ce_calc.get_energy()),
            "singl_eng": np.zeros(n_singlets, dtype=np.float64),
            "counter": 0
        }

    def reset(self):
        """
        Resets all variables to zero
        """
        self.quantities["singlets"][:] = 0.0
        self.quantities["singlets_sq"][:] = 0.0
        self.quantities["energy"].clear()
        self.quantities["energy_sq"].clear()
        self.quantities["singl_eng"][:] = 0.0
        self.quantities["counter"] = 0

    def __call__(self, system_changes):
        """
        Updates all SGC parameters

        :param list system_changes: Last changes to the system
        """
        self.quantities["counter"] += 1
        new_singlets = self.ce_calc.get_singlets()

        if self.recycle_waste:
            avg_singl = np.zeros_like(self.singlets)
            avg_sq = np.zeros_like(self.singlets)
            avg_corr = np.zeros_like(self.singlets)
            for i in range(len(new_singlets)):
                singl = np.array([self.mc.current_singlets[i],
                                  new_singlets[i]])

                avg_singl[i] = waste_recycled_average(singl,
                                                      self.mc.last_energies,
                                                      self.mc.T)
                avg_sq[i] = waste_recycled_average(singl**2,
                                                   self.mc.last_energies,
                                                   self.mc.T)

                singl_en = np.array(
                    [self.mc.current_singlets[i] * self.mc.last_energies[0],
                     new_singlets[i] * self.mc.last_energies[1]])

                avg_corr[i] = waste_recycled_average(singl_en,
                                                     self.mc.last_energies,
                                                     self.mc.T)

            E = waste_recycled_average(self.mc.last_energies,
                                       self.mc.last_energies, self.mc.T)
            E_sq = waste_recycled_average(self.mc.last_energies**2,
                                          self.mc.last_energies, self.mc.T)
            self.quantities["energy"] += E
            self.quantities["energy_sq"] += E_sq
            self.quantities["singlets"] += avg_singl
            self.quantities["singlets_sq"] += avg_sq
            self.quantities["singl_eng"] += avg_corr
        else:
            self.quantities["singlets"] += new_singlets
            self.quantities["singlets_sq"] += new_singlets**2
            self.quantities["energy"] += self.mc.current_energy_without_vib()
            self.quantities["energy_sq"] += \
                self.mc.current_energy_without_vib()**2
            self.quantities["singl_eng"] += \
                new_singlets*self.mc.current_energy_without_vib()

    @property
    def energy(self):
        return self.quantities["energy"]

    @property
    def energy_sq(self):
        return self.quantities["energy_sq"]

    @property
    def singlets(self):
        return self.quantities["singlets"]

    @property
    def singl_eng(self):
        return self.quantities["singl_eng"]

    @property
    def counter(self):
        return self.quantities["counter"]


class Snapshot(MCObserver):
    """
    Store a snapshot in a trajectory file

    :param str trajfile: Filename of the trajectory file
    :param Atoms atoms: Instance of the atoms objected modofied by the MC object
    """

    def __init__(self, trajfile="default.traj", atoms=None):
        super(Snapshot, self).__init__()
        self.name = "Snapshot"
        if not trajfile.endswith(".traj"):
            msg = "This object stores all images in a trajectory file. "
            msg += "File extension should be .traj"
            raise ValueError(msg)
        if atoms is None:
            raise ValueError("No atoms object given!")
        self.atoms = atoms
        self.traj = TrajectoryWriter(trajfile, mode="a")
        self.fname = trajfile

    def __call__(self, system_changes):
        self.traj.write(self.atoms)


class NetworkObserver(MCObserver):
    """
    Track networks of atoms being connected by one of the pair interactions

    :param CE calc: Instance of the CE calculator
    :param list cluster_name: Name of the cluster (has to be a pair interaction)
        Example [c2_5p72]
    :param list element: Element tracked. If a network is defined by Mg atoms
        connected via some pair cluster this is Mg
    :param int nbins: Number of bins used to produce statistics over the
        distribution of cluster sizes
    :param Intracomm mpicomm: MPI communicator
    """

    def __init__(self, calc=None, cluster_name=None, element=None, nbins=30,
                 mpicomm=None):
        if calc is None:
            raise ValueError("No calculator given. " +
                             "Has to be a CE calculator (with C++ support)")
        if cluster_name is None:
            raise ValueError("No cluster name given!")
        if element is None:
            raise ValueError("No element given!")
        # self.fast_cluster_tracker = ce_updater.ClusterTracker(calc.updater,
        #                                                       cluster_name,
        #                                                       element)
        self.calc = calc
        self.cluster_name = cluster_name
        self.element = element
        self.fast_cluster_tracker = PyClusterTracker(calc.updater,
                                                     cluster_name,
                                                     element)
        super(NetworkObserver, self).__init__()
        self.name = "NetworkObserver"
        self.calc = calc
        self.res = {"avg_size": 0.0,
                    "avg_size_sq": 0.0,
                    "number_of_clusters": 0}
        self.max_size = 0
        self.indx_max_cluster = []
        self.atoms_max_cluster = None
        self.n_calls = 0
        self.n_atoms_in_cluster = 0
        self.mpicomm = mpicomm

        # Count the number of atoms of the element type being tracked
        n_atoms = 0
        for atom in self.calc.atoms:
            if atom.symbol == element:
                n_atoms += 1
        self.max_size_hist = n_atoms
        self.num_clusters = 0
        self.nbins = nbins
        self.size_histogram = np.zeros(self.nbins)

    def __reduce__(self):
        args = (self.calc, self.cluster_name, self.element, self.nbins)
        return (self.__class__, args)

    def __call__(self, system_changes):
        """
        Collect information about atomic clusters in the system

        :param list system_changes: Last changes to the system
        """
        self.n_calls += 1
        self.fast_cluster_tracker.find_clusters()

        new_res = self.fast_cluster_tracker.get_cluster_statistics_python()
        for key in self.res.keys():
            self.res[key] += new_res[key]

        self.update_histogram(new_res["cluster_sizes"])
        self.n_atoms_in_cluster += np.sum(new_res["cluster_sizes"])
        if new_res["max_size"] > self.max_size:
            self.max_size = new_res["max_size"]
            self.atoms_max_cluster = self.calc.atoms.copy()
            clust_indx = \
                self.fast_cluster_tracker.atomic_clusters2group_indx_python()
            self.indx_max_cluster = clust_indx
            self.num_clusters = len(new_res["cluster_sizes"])

    def has_minimal_connectivity(self):
        return self.fast_cluster_tracker.has_minimal_connectivity()

    def retrieve_clusters_from_scratch(self):
        """Retrieve the all the clusters from scratch."""
        self.fast_cluster_tracker.find_clusters()

    def update_histogram(self, sizes):
        """
        Update the histogram

        :param list sizes: Cluster sizes
        """
        for size in sizes:
            if size >= self.max_size_hist:
                continue
            indx = int(self.nbins * float(size) / self.max_size_hist)
            self.size_histogram[indx] += 1

    def reset(self):
        """Reset the observer."""
        for key in self.res.keys():
            self.res[key] = 0

        self.max_size = 0
        self.indx_max_cluster = []
        self.atoms_max_cluster = None
        self.n_calls = 0
        self.n_atoms_in_cluster = 0
        self.num_clusters = 0

    def get_atoms_with_largest_cluster(self, prohibited_symbols=[]):
        """
        Return the atoms object which had the largest cluster and change the
        element of the atoms in the cluster to *highlight_element*

        :param list prohibited_symbols: symbols that can not be used to
            highlight

        :return: Atoms object with clusters highlighted
        :rtype: Atoms
        """
        if self.atoms_max_cluster is None:
            from ase.build import bulk
            print("No clusters was detected!")
            return bulk("Al")
        group_indx_count = self.get_cluster_count()

        elems_in_atoms_obj = []
        for atom in self.atoms_max_cluster:
            if atom.symbol not in elems_in_atoms_obj:
                elems_in_atoms_obj.append(atom.symbol)

        current_highlight_element = 0
        high_elms = \
            self.generate_highlight_elements_from_size(group_indx_count,
                                                       prohibited_symbols)
        for key, value in group_indx_count.items():
            if value <= 3:
                continue
            for i, indx in enumerate(self.indx_max_cluster):
                if indx == key:
                    self.atoms_max_cluster[i].symbol = high_elms[key]
            current_highlight_element += 1
        return self.atoms_max_cluster

    def generate_highlight_elements_from_size(self, group_indx_count,
                                              prohibited_symbols):
        """Create list of highlight elements based on the group index count.

        :param dict group_indx_count: Number of atoms in each cluster
        :param list prohibited_symbols: Symbols that cannot be used to
            highlight
        """
        tup = []
        for key, value in group_indx_count.items():
            if value <= 3:
                continue
            tup.append((value, key))

        tup.sort()
        tup = tup[::-1]
        max_n_elems = len(highlight_elements)
        highlist = {clst[1]: highlight_elements[i % max_n_elems] for i, clst
                    in enumerate(tup)}
        highlist = {}
        counter = 0
        for clst in tup:
            while highlight_elements[counter] in prohibited_symbols:
                counter += 1
            highlist[clst[1]] = highlight_elements[counter % max_n_elems]
        return highlist

    def get_cluster_count(self):
        """
        Counts the number of atoms in each clusters

        :return: Number of atoms in each cluster
        :rtype: dict
        """
        group_indx_count = {}
        for indx in self.indx_max_cluster:
            if indx in group_indx_count.keys():
                group_indx_count[indx] += 1
            else:
                group_indx_count[indx] = 1
        return group_indx_count

    def get_indices_of_largest_cluster(self):
        """
        Return the indices of the largest cluster

        :return: Indices of the atoms in the largest cluster
        :rtype: list of int
        """
        group_indx_count = self.get_cluster_count()
        max_id = 0
        max_size = 0
        for key, value in group_indx_count.items():
            if value > max_size:
                max_size = value
                max_id = key
        return [i for i, indx in enumerate(self.indx_max_cluster)
                if indx == max_id]

    @staticmethod
    def flatten_nested_list(nested):
        flattened = []
        for sublist in nested:
            flattened += sublist
        return flattened

    def get_indices_of_largest_cluster_with_neighbours(self):
        """Return the indices of the largest cluster+their neighboirs
        :return: Indices of atoms in the cluster
        :rtype: list of int
        """
        indices = self.get_indices_of_largest_cluster()
        neighbour_indices = []
        for root in indices:
            for cname in self.cluster_name:
                for info in self.calc.BC.cluster_info:
                    if cname not in info.keys():
                        continue
                    members = info[cname]["indices"]
                    for i in NetworkObserver.flatten_nested_list(members):
                        tindx = self.calc.BC.trans_matrix[root][i]
                        neighbour_indices.append(tindx)
        indices += neighbour_indices
        return list(set(indices))

    def collect_stat_MPI(self):
        """Collect the statistics from MPI."""
        if self.mpicomm is None:
            return
        recv_buf = np.zeros_like(self.size_histogram)
        self.mpicomm.Allreduce(self.size_histogram, recv_buf)
        self.size_histogram[:] = recv_buf[:]

        # Find the maximum cluster
        max_size = self.mpicomm.gather(self.max_size, root=0)
        rank = self.mpicomm.Get_rank()
        if rank == 0:
            self.max_size = np.max(max_size)
        self.max_size = self.mpicomm.bcast(self.max_size, root=0)

        if rank == 0:
            msg = "Waring! The MPI collection of results for the "
            msg += "NetworkObserver is incomplete. The histogram is correctly "
            msg += "collected and the maximum cluster size. Entries received "
            msg += "by get_statisttics() is not collected yet."
            print(msg)

    def get_current_cluster_info(self):
        """Return the info dict for the current state."""
        self.fast_cluster_tracker.find_clusters()
        return self.fast_cluster_tracker.get_cluster_statistics_python()

    def get_statistics(self):
        """Compute network size statistics.

        :return: Statistics about atomic clusters in the system
        :rtype: dict
        """
        self.collect_stat_MPI()

        stat = {}
        if self.res["number_of_clusters"] == 0:
            stat["avg_size"] = 0
            avg_sq = 0
        else:
            stat["avg_size"] = self.res["avg_size"]
            stat["avg_size"] /= self.res["number_of_clusters"]
            avg_sq = self.res["avg_size_sq"] / self.res["number_of_clusters"]
        stat["std"] = np.sqrt(avg_sq - stat["avg_size"]**2)
        stat["max_size"] = self.max_size
        stat["n_atoms_in_cluster"] = self.n_atoms_in_cluster
        stat["number_of_clusters"] = int(self.res["number_of_clusters"])
        if self.max_size_hist == 0:
            stat["frac_atoms_in_cluster"] = 0.0
        else:
            stat["frac_atoms_in_cluster"] = float(self.n_atoms_in_cluster)
            stat["frac_atoms_in_cluster"] /= self.n_calls * self.max_size_hist
        return stat

    def get_size_histogram(self):
        """Return the size histogram and the corresponding size.

        :return: Sizes and corresponding occurence rate
        :rtype: 1D numpy array, 1D numpy array
        """
        x = np.linspace(3, self.max_size_hist, self.nbins)
        return x, self.size_histogram

    def surface(self):
        """
        Computes the surface of a cluster
        """
        return self.fast_cluster_tracker.surface_python()


class SiteOrderParameter(MCObserver):
    """
    Class that can be used to detect phase transitions.

    It monitors the average number of sites that are different from the initial
    value.

    :param Atoms atoms: Atoms object
    :param mpicomm: MPI communicator
    :type mpicomm: Intracomm or None
    """

    def __init__(self, atoms, mpicomm=None):
        self.atoms = atoms
        self.orig_nums = self.atoms.get_atomic_numbers()
        self.avg_num_changed = 0
        self.avg_num_changed_sq = 0
        self.num_calls = 0
        self.current_num_changed = 0
        self.site_changed = np.zeros(len(self.atoms), dtype=np.uint8)
        self.mpicomm = mpicomm

    def _check_all_sites(self):
        """Check if symbols have changed on all sites."""
        nums = self.atoms.get_atomic_numbers()
        self.current_num_changed = np.count_nonzero(nums != self.orig_nums)
        self.site_changed = (nums != self.orig_nums)

    def reset(self):
        """Resets the tracked data. (Not the original symbols array)."""
        self.avg_num_changed = 0
        self.avg_num_changed_sq = 0
        self.num_calls = 0
        self.current_num_changed = 0
        self._check_all_sites()

    def __call__(self, system_changes):
        """Get a new value for the order parameter.

        :param list system_changes: Last changes to the system
        """

        self.num_calls += 1
        assert self.current_num_changed < len(self.atoms)

        # The point this function is called the atoms object is already
        # updated
        for change in system_changes:
            indx = change[0]
            if self.site_changed[indx]:
                if self.atoms[indx].number == self.orig_nums[indx]:
                    self.current_num_changed -= 1
                    self.site_changed[indx] = False
            else:
                if self.atoms[indx].number != self.orig_nums[indx]:
                    self.current_num_changed += 1
                    self.site_changed[indx] = True
        self.avg_num_changed += self.current_num_changed
        self.avg_num_changed_sq += self.current_num_changed**2

    def get_average(self):
        """Get the number of sites different from the ground state.

        :return: Average and standard deviation of the order parameters
        :rtype: float, float
        """
        average = float(self.avg_num_changed)/self.num_calls
        average_sq = float(self.avg_num_changed_sq)/self.num_calls

        if self.mpicomm is not None:
            size = self.mpicomm.Get_size()
            send_buf = np.zeros(2)
            send_buf[0] = average
            send_buf[1] = average_sq
            recv_buf = np.zeros(2)
            self.mpicomm.Allreduce(send_buf, recv_buf)
            average = recv_buf[0]/size
            average_sq = recv_buf[1]/size
        var = average_sq - average**2

        # If variance is close to zero it can in some cases by
        # slightly negative. Add a safety check for this
        if var < 0.0:
            var = 0.0
        return average, np.sqrt(var)


class EnergyEvolution(MCObserver):
    def __init__(self, mc_obj):
        self.mc = mc_obj
        self.energies = []
        MCObserver.__init__(self)
        self.name = "EnergyEvolution"

    def __call__(self, system_changes):
        """Append the current energy to the MC object."""
        self.energies.append(self.mc.current_energy_without_vib())

    def reset(self):
        """Reset the history."""
        self.energies = []


class EnergyHistogram(MCObserver):
    def __init__(self, mc_obj, buffer_size=100000, n_bins=100):
        self.mc = mc_obj
        self.buffer = np.zeros(buffer_size)
        self.n_bins = n_bins
        MCObserver.__init__(self)
        self._next = 0
        self._histogram = None
        self.Emin = None
        self.EMax = None
        self.sample_in_buffer = True

    def __call__(self, system_changes):
        E = self.mc.current_energy_without_vib()

        if self.sample_in_buffer:
            self.buffer[self._next] = E
            self._next += 1
            if self._buffer_is_full():
                self._on_buffer_full()
        else:
            indx = self._get_indx(E)
            self._histogram[indx] += 1

    def _buffer_is_full(self):
        """Return True if the buffer is full."""
        return self._next >= len(self.buffer)

    def _on_buffer_full(self):
        """Initialize the histogram and create a histogram."""
        self.Emin = np.min(self.buffer)
        self.Emax = np.max(self.buffer)
        self._histogram = np.zeros(len(self.n_bins))
        for e in self.buffer:
            indx = self._get_indx(e)
            self._histogram[indx] += 1

        # After initialization we don't need to buffer the energies any more
        self.sample_in_buffer = False

    def _get_indx(self, E):
        """Return the index in the histogram corresponding to E."""
        if self.Emin is None or self.Emax is None:
            raise RuntimeError("This function should never be called before "
                               "the histogram has been updated at least once!")
        return int((E-self.Emin)*(self.n_bins - 1)/(self.Emax - self.Emin))

    @property
    def histogram(self):
        if self._histogram is None:
            self._on_buffer_full()
        return self._histogram


class MCBackup(MCObserver):
    """Class that makes backup of the current MC object.

    :param Montecarlo mc_obj: Monte Carlo object
    :param str backup_file: Filename where backup will be written. Note that
        the content of this file will be overwritten everytime.
    :param str db_name: Database name. If given, results will be written to 
        a table
    :param str db_tab_name: Name of table in the database
    :param int db_id: ID in the database. If None, a new entry will be created
    """

    def __init__(self, mc_obj, backup_file="montecarlo_backup.pkl", db_name="",
                 db_tab_name="mc_backup", db_id=None):
        self.mc_obj = mc_obj
        self.backup_file = self._include_rank_in_filename(backup_file)
        MCObserver.__init__(self)
        self.name = "MCBackup"
        self.db_name = db_name
        self.db_id = None
        self.db_tab_name = db_tab_name

    def _include_rank_in_filename(self, fname):
        """Include the current rank in the filename if nessecary."""
        size = num_processors()
        if size > 1:
            # We have to include the rank in the filename to avoid problems
            rank = mpi_rank()
            prefix = fname.rpartition(".")[0]
            return prefix + "_rank{}.pkl".format(rank)
        return fname

    def __call__(self, system_changes):
        """Write a copy of the Monte Carlo object to file."""
        self.mc_obj.save(self.backup_file)
        if self.db_name != "":
            import dataset
            thermo = self.mc_obj.get_thermodynamic()
            db = dataset.connect("sqlite:///{}".format(self.db_name))
            tab = db[self.db_tab_name]
            if self.db_id is None:
                # This shoud be a new entry
                self.db_id = tab.insert(thermo)
            else:
                # Entry alread exists. Update that one.
                thermo["id"] = self.db_id
                tab.update(thermo, ["id"])


class BiasPotentialContribution(MCObserver):
    def __init__(self, mc=None, buffer_size=10000, n_bins=100):
        self.mc = mc
        self.buffer = np.zeros(buffer_size)
        self.hist = np.zeros(n_bins)
        self.hist_max = None
        self.hist_min = None
        self.buffer_indx = 0

    def __call__(self, system_changes):
        diff = self.mc.new_bias_energy - self.mc.bias_energy
        self.buffer[self.buffer_indx] = diff
        self.buffer_indx += 1

        if self.buffer_indx == len(self.buffer):
            self._update_histogram()

    def _update_histogram(self):
        """Updates the histogram with the current buffer."""
        if self.hist_max is None:
            self.hist_max = np.max(self.buffer)
            self.hist_min = np.min(self.buffer)
            hist_range = self.hist_max - self.hist_min

            # We double the range in case the first
            # buffer did not cover all cases
            self.hist_max += hist_range/2.0
            self.hist_min -= hist_range/2.0

        hist_indx = (self.buffer - self.hist_min)*len(self.hist) \
                    /(self.hist_max - self.hist_min)
        
        for indx in hist_indx:
            if indx < len(self.hist) and indx >= 0:
                self.hist[indx] += 1

    def save(self, fname="bias_potential_hist.csv"):
        """Store the histogram to a text file."""
        if self.hist_max is None:
            # There is not histogram
            return
        x = np.linspace(self.hist_min, self.hist_max, len(self.hist))
        data = np.vstack((x, self.hist))
        np.savetxt(fname, data.T, delimiter=",")
        print("Histogram data written to {}".format(fname))


class InertiaTensorObserver(MCObserver):
    def __init__(self, atoms=None, cluster_elements=None):
        self.pos = atoms.get_positions()
        self.cluster_elements = cluster_elements
        self.com = np.zeros(3)
        self.inertia = np.zeros((3, 3))
        self.num_atoms = 0
        self._init_com_and_inertia(atoms)
        self.old_com = None
        self.old_inertia = None

    def _init_com_and_inertia(self, atoms):
        """Initialize the center of mass and the inertia."""
        self.num_atoms = 0
        for atom in atoms:
            if atom.symbol in self.cluster_elements:
                self.com += atom.position
                self.inertia += np.outer(atom.position, atom.position)
                self.num_atoms += 1

        if self.num_atoms == 0:
            raise RuntimeError("No cluster elements are present in the "
                               "Atoms object provided!")
        self.com /= self.num_atoms
        self.inertia -= self.num_atoms*np.outer(self.com, self.com)

    def __call__(self, system_changes):
        """Update the inertia tensor."""
        d_com = np.zeros(3)
        d_I = np.zeros((3, 3))
        for change in system_changes:
            if change[1] in self.cluster_elements and change[2] in self.cluster_elements:
                continue

            x = self.pos[change[0], :]
            if change[2] in self.cluster_elements:
                d_com += x
                d_I += np.outer(x, x)
            elif change[1] in self.cluster_elements:
                d_com -= x
                d_I -= np.outer(x, x)
        
        d_com /= self.num_atoms

        d_I -= self.num_atoms*(np.outer(d_com, self.com) + np.outer(self.com, d_com) + np.outer(d_com, d_com))
        self.old_inertia = self.inertia.copy()
        self.old_com = self.com.copy()

        self.com += d_com
        self.inertia += d_I

    def undo_last(self):
        """Undo the last update."""
        if self.old_inertia is None:
            return
        self.inertia = self.old_inertia
        self.com = self.old_com

            

class PairObserver(MCObserver):
    """Tracking the average number of pairs within a cutoff"""
    def __init__(self, atoms, cutoff=4.0, elements=[]):
        from ase.neighborlist import neighbor_list
        self.atoms = atoms
        self.cutoff = cutoff
        self.elements = elements
        first_indx, second_indx, self.dist = neighbor_list("ijd", atoms, cutoff)

        # neighbors
        self.neighbors = [[] for _ in range(len(self.atoms))]
        for i1, i2 in zip(first_indx, second_indx):
            self.neighbors[i1].append(i2)

        # Count how many pairs inside cutoff
        self.num_pairs = self.num_pairs_brute_force()
        self.avg_num_pairs = 0
        self.num_calls = 0
        self.symbols = [atom.symbol for atom in self.atoms]

    def __call__(self, system_changes):
        # Update how many pairs there are present
        # at this point the atoms object is already 
        num_new_pairs = 0
        for change in system_changes:
            if change[1] in self.elements and change[2] not in self.elements:
                neighbors = self.neighbors[change[0]]
                pairs_in_site = len([self.symbols[indx] for indx in neighbors
                                     if self.symbols[indx] in self.elements])

                # We loose some pairs (factor 2 due to double counting)
                num_new_pairs -= 2*pairs_in_site

            elif change[1] not in self.elements and change[2] in self.elements:
                neighbors = self.neighbors[change[0]]
                pairs_in_site = len([self.symbols[indx] for indx in neighbors
                                     if self.symbols[indx] in self.elements])
                # Add pairs (factor 2 due to double counting)
                num_new_pairs += 2*pairs_in_site
            self.symbols[change[0]] = change[2]

        self.num_pairs += num_new_pairs
        self.avg_num_pairs += float(self.num_pairs)/len(self.atoms)
        self.num_calls += 1

    def reset(self):
        self.num_calls = 0
        self.avg_num_pairs = 0

    def symbols_is_synced(self):
        """Sanity check to ensure that the symbols array is syncronized."""
        symbs_atoms = [atom.symbol for atom in self.atoms]
        return symbs_atoms == self.symbols

    def num_pairs_brute_force(self):
        """Calculate the number pairs by brute force loop."""
        num_pairs = 0
        for i1 in range(len(self.neighbors)):
            if self.atoms[i1].symbol in self.elements:
                for i2 in self.neighbors[i1]:
                    if self.atoms[i2].symbol in self.elements:
                        num_pairs += 1
        return num_pairs

    @property
    def mean_number_of_pairs(self):
        if self.num_calls == 0:
            return 0
        return self.avg_num_pairs/self.num_calls





