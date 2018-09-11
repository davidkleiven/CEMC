from cemc.mcmc import Montecarlo
from cemc.mcmc import NetworkObserver
import numpy as np


class FixedNucleusMC(Montecarlo):
    """
    Class that performs Monte Carlo where one element type is constrained
    to be in a certain network of atoms. This can be useful if one for instance
    would like to study the energetics of different shapes of the network

    See :py:class:`cemc.mcmc.Montecarlo`

    :Keyword arguments:
        * *size* Network size
        * *network_element* Element in the network see
        :py:class:`cemc.mcmc.mc_observers.NetworkObserver`
        * *cluster_name* Name of clusters see
        :py:class:`cemc.mcmc.mc_observers.NetworkObserver`
        * In addition all keyword arguments of
        :py:class:`cemc.mcmc.Montecarlo` can be given
    """

    def __init__(self, atoms, T, **kwargs):
        self.network_element = kwargs.pop("network_element")
        self.network_name = kwargs.pop("network_name")

        if type(self.network_element) == str:
            self.network_element = [self.network_element]
        if type(self.network_name) == str:
            self.network_name = [self.network_name]

        self.size = None
        super(FixedNucleusMC, self).__init__(atoms, T, **kwargs)

        self.network = self._init_networks()
        self.bc = self.atoms._calc.BC
        self.network_clust_indx = self.find_cluster_indx()

    def _init_networks(self):
        """Initialize the network observers."""
        network = NetworkObserver(
            calc=self.atoms._calc, cluster_name=self.network_name,
            element=self.network_element)
        return network

    def get_translated_indx(self, ref_indx, clst_indx):
        return self.bc.trans_matrix[ref_indx][clst_indx]

    @property
    def num_clusters(self):
        self.network.reset()
        self.network(None)
        stat = self._get_network_statistics()
        self.network.reset()
        return stat["number_of_clusters"]

    def find_cluster_indx(self):
        """
        Find the cluster indices corresponding to the current network name
        """
        network_indx = []
        for name in self.network_name:
            clusters = self.bc.cluster_info_by_name(name)
            for clst in clusters:
                flattened = [item[0] for item in clst["indices"]]
                network_indx += flattened
        return np.unique(network_indx)

    def _get_trial_move(self):
        """Generate a trial move."""
        from random import choice, shuffle
        ref_element = choice(self.network_element)
        Na = len(self.atoms_indx[ref_element])
        # max_rand = len(self.network_clust_indx)
        self.selected_a = np.random.randint(low=0, high=Na)
        self.rand_a = self.atoms_indx[ref_element][self.selected_a]

        # Shuffle the network indices
        shuffle(self.network_clust_indx)
        symb_b = ref_element
        while symb_b == ref_element:
            new_ref_element = choice(self.network_element)
            Nb = len(self.atoms_indx[new_ref_element])
            temp_rand_num_b = np.random.randint(low=0, high=Nb)
            ref_indx = self.atoms_indx[new_ref_element][temp_rand_num_b]
            for indx in self.network_clust_indx:
                self.rand_b = self.bc.trans_matrix[ref_indx][indx]
                symb_b = self.atoms[self.rand_b].symbol

        # TODO: Make the next line more efficient (avoid search!)
        self.selected_b = self.atoms_indx[symb_b].index(self.rand_b)

        symb_a = self.atoms[self.rand_a].symbol
        assert symb_a == ref_element
        system_changes = [(self.rand_a, symb_a, symb_b),
                          (self.rand_b, symb_b, symb_a)]
        return system_changes

    def _get_network_statistics(self):
        """Retrieve statistics from all networks."""
        return self.network.get_statistics()

    def _size_ok(self, stat):
        """Check if the sizes are OK."""
        full_size = 0
        for st in stat:
            full_size += st["max_size"]

        if self.size is None:
            self.size = full_size
        return full_size == self.size

    def _accept(self, system_changes):
        """Accept trial move."""
        move_accepted = Montecarlo._accept(self, system_changes)

        if self.num_clusters != 1:
            return False
        return move_accepted

    def _check_nucleation_site_exists(self):
        """Check that at least one nucleation site exists."""
        for atom in self.atoms:
            if atom.symbol == self.network_element[0]:
                return
        elm = self.network_element[0]
        raise ValueError("At least one element of {} "
                         "has to be present".format(elm))

    def _get_initial_site(self):
        """Get an initial site in the correct symm group."""
        element = self.network_element[0]
        indx = self.atoms_indx[element][0]
        return indx

    def _grow_cluster(self, elements):
        """Grow a cluster of a certain size."""
        from random import choice, shuffle
        all_elems = []
        at_count = self.count_atoms()
        unique_solute_elements = list(elements.keys())
        for symb, num in elements.items():
            if symb in at_count.keys():
                num_insert = num - at_count[symb]
            else:
                num_insert = num

            if num_insert < 0:
                raise ValueError("There are too many solute atoms in the "
                                 "system for the requested cluster size!\n"
                                 "Requested: {}\n"
                                 "At. count: {}".format(elements, at_count))

            for _ in range(num_insert):
                all_elems.append(symb)

        ref_site = self._get_initial_site()
        inserted_indices = []
        while all_elems:
            shuffle(self.network_clust_indx)

            element = choice(all_elems)
            found_place_to_insert = False
            for net_indx in self.network_clust_indx:
                indx = self.bc.trans_matrix[ref_site][net_indx]
                old_symb = self.atoms[indx].symbol
                if old_symb in unique_solute_elements:
                    continue
                system_changes = (indx, old_symb, element)

                if self._no_constraint_violations([system_changes]):
                    self.atoms.get_calculator().update_cf(system_changes)
                    all_elems.remove(element)
                    found_place_to_insert = True
                    inserted_indices.append(indx)
                    break
            if found_place_to_insert:
                # Store the site that was inserted as a candidate to
                # the next reference site
                new_candidate_ref_site = indx
            else:
                # Did not manage to insert a new element with the current
                # reference site, so try another reference site
                ref_site = new_candidate_ref_site
        self._build_atoms_list()
        self.atoms.get_calculator().clear_history()

        # Verify that the construction was fine
        at_count = self.count_atoms()
        for k, target_num in elements.items():
            if at_count[k] != target_num:
                raise ValueError("Inconsistent size!\n"
                                 "Should be: {}\n"
                                 "Contains: {}".format(elements, at_count))

        self.network.reset()
        self.network(None)
        stat = self.network.get_statistics()

        if stat["number_of_clusters"] != 1:
            print("Cluster statistics:")
            print(stat)
            indices = []
            for atom in self.atoms:
                if atom.symbol in unique_solute_elements:
                    indices.append(atom.index)
            cluster = self.atoms[indices]
            from ase.io import write
            fname = "invalide_cluster{}.xyz".format(self.rank)
            write(fname, cluster)
            raise RuntimeError("There are {} clusters present! "
                               "Initial structure written to {}\n"
                               "".format(stat["number_of_clusters"], fname))

    def get_atoms(self, atoms=None, prohib_elem=[]):
        """
        Return the atoms object with the clusters highlighted
        """
        if atoms is None:
            atoms = self.atoms
        else:
            symbs = [atom.symbol for atom in atoms]
            self.atoms._calc.set_symbols(symbs)
            atoms = self.atoms

        self.network.reset()
        indices = self.network.get_indices_of_largest_cluster()

        ref_atom = indices[0]
        pos = atoms.get_positions()
        mic_dists = atoms.get_distances(ref_atom, indices, mic=True,
                                        vector=True)
        com = pos[ref_atom, :] + np.mean(mic_dists, axis=0)
        cell = atoms.get_cell()
        center = 0.5 * (cell[0, :] + cell[1, :] + cell[2, :])
        atoms.translate(center-com)
        atoms.wrap()
        return atoms, atoms[indices]

    def runMC(self, steps=100000, init_cluster=True, elements={}):
        """
        Run Monte Carlo for fixed nucleus size

        :param steps: Number of Monte Carlo steps
        :param init_cluster: If True initialize a cluster, If False it is
            assumed that a cluster of the correct size already exists in the
            system
        """
        if init_cluster:
            self._check_nucleation_site_exists()
            self._grow_cluster(elements)
        step = 0
        while step < steps:
            step += 1
            self._mc_step()
        accpt_rate = float(self.num_accepted) / self.current_step
        print("Acceptance rate: {}".format(accpt_rate))
