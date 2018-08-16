import matplotlib as mpl
from matplotlib import pyplot as plt
import copy
import numpy as np
from cemc.ce_updater import ce_updater
from ase.io.trajectory import TrajectoryWriter
from cemc.mcmc.averager import Averager
from cemc.mcmc.util import waste_recycled_average

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
highlight_elements = ["Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
                      "Al", "Si", "P", "S", "Cl", "Ar"]


class MCObserver(object):
    """Base class for all MC observers."""

    def __init__(self):
        self.name = "GenericObserver"

    def __call__(self, system_changes):
        """
        Gets information about the system changes and can perform some action

        :param system_changes: List of system changes if indx 23 changed
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

    :param ce_calc: Instance of the CE calculator attached to the atoms object
    """

    def __init__(self, ce_calc):
        self.cf = []
        self.ce_calc = ce_calc
        self.name = "CorrelationFunctionTracker"

    def __call__(self, system_changes):
        """Update the correlation functions."""
        self.cf.append(copy.deepcopy(self.ce_calc.cf))

    def plot_history(self, max_size=10):
        """Plot history (only if history is tracked)."""
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
    """Compute the average value of all the ECIs."""

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

        for key, value in self.ce_calc.eci.items():
            if key.startswith("c2_"):
                self.cf[key] = 0.0
                self.cf_squared[key] = 0.0

    def __call__(self, system_changes):
        """Update correlation functions."""
        new_cf = self.ce_calc.updater.get_cf()
        self.n_entries += 1
        for key in self.cf.keys():
            self.cf[key] += new_cf[key]
            self.cf_squared[key] += new_cf[key]**2

    def get_average(self):
        """Returns the average."""
        avg_cf = copy.deepcopy(self.cf)
        for key in avg_cf.keys():
            avg_cf[key] /= self.n_entries
        return avg_cf

    def get_std(self):
        """Return the standard deviation."""
        std_cf = {key: 0.0 for key in self.cf.keys()}
        for key in self.cf.keys():
            std_cf[key] = np.sqrt(self.cf_squared[key] / self.n_entries
                                  - (self.cf[key] / self.n_entries)**2)
        return std_cf


class LowestEnergyStructure(MCObserver):
    """
    Observer that tracks the lowest energy state visited
    during an MC run

    :param ce_calc: Instance of the CE calculator
    :param mc_obj: Monte Carlo object
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

    :param ce_calc: CE calculator
    :param mc_obj: Instance of the Monte Carlo object
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
        """
        self.quantities["counter"] += 1
        new_singlets = np.zeros_like(self.singlets)
        self.ce_calc.get_singlets(new_singlets)

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

    :param trajfile: Filename of the trajectory file
    :param atoms: Instance of the atoms objected modofied by the MC object
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

    :param calc: Instance of the CE calculator
    :param cluster_name: Name of the cluster (has to be a pair interaction)
        Example c2_5p72
    :param element: Element tracked. If a network is defined by Mg atoms
        connected via some pair cluster this is Mg
    :param nbins: Number of bins used to produce statistics over the
        distribution of cluster sizes
    :param mpicomm: MPI communicator
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
        self.fast_cluster_tracker = ce_updater.ClusterTracker(calc.updater,
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

    def __call__(self, system_changes):
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

    def update_histogram(self, sizes):
        """
        Update the histogram

        :param sizes: Cluster sizes
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
        """
        if self.atoms_max_cluster is None:
            print("No clusters was detected!")
            return None
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
        """Create list of highlight elements based on the group index count."""
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

    def get_statistics(self):
        """Compute network size statistics."""
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
        """Return the size histogram and the corresponding size."""
        x = np.linspace(3, self.max_size_hist, self.nbins)
        return x, self.size_histogram

    def grow_cluster(self, size):
        """
        Grow a cluster of the given size

        :param size: Target size
        """
        self.fast_cluster_tracker.grow_cluster(size)

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
        """Get the number of sites different from the ground state."""
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
