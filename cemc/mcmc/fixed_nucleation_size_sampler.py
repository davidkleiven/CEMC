from cemc.mcmc import Montecarlo
from cemc.mcmc import NetworkObserver
import numpy as np


class FixedNucleusMC(Montecarlo):
    """
    Class that performs Monte Carlo where one element type is constrained
    to be in a certain network of atoms. This can be useful if one for instance
    would like to study the energetics of different shapes of the network

    See :py:class:`cemc.mcmc.Montecarlo`

    :param list network_element: List of elements in the cluster
    :param list cluster_name: List of cluster names that can links atoms
        in a cluster
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
        self.bc = self.atoms.get_calculator().BC
        self.network_clust_indx = self.find_cluster_indx()
        self.initial_num_atoms_in_cluster = 0
        self.attach(self.network)

    def _init_networks(self):
        """Initialize the network observers."""
        network = NetworkObserver(
            calc=self.atoms._calc, cluster_name=self.network_name,
            element=self.network_element)
        return network

    def get_translated_indx(self, ref_indx, clst_indx):
        """Get the translated index

        :param int ref_indx: Reference index
        :param int clst_indx: Relative index

        :return: The translated index
        :rtype: int
        """
        return self.bc.trans_matrix[ref_indx][clst_indx]

    @property
    def num_clusters(self):
        self.network.reset()
        self.network(None)
        stat = self._get_network_statistics()
        self.network.reset()
        return stat["number_of_clusters"]

    @property
    def num_atoms_in_cluster(self):
        self.network.reset()
        self.network(None)
        stat = self._get_network_statistics()
        self.network.reset()
        return stat["n_atoms_in_cluster"]

    @property
    def cluster_stat(self):
        self.network.reset()
        self.network(None)
        stat = self._get_network_statistics()
        self.network.reset()
        return stat

    def find_cluster_indx(self):
        """
        Find the cluster indices corresponding to the current network name

        :return: List with unique network indices (relative index)
        :rtype: list of int
        """
        network_indx = []
        for name in self.network_name:
            clusters = self.bc.cluster_info_by_name(name)
            for clst in clusters:
                flattened = [item[0] for item in clst["indices"]]
                network_indx += flattened
        return np.unique(network_indx)

    def _get_trial_move(self):
        """Generate a trial move.

        :return: Proposed move
        :rtype: list of tuples
        """
        from random import choice, shuffle
        ref_element = choice(self.network_element)
        rand_a = self.atoms_tracker.get_random_indx_of_symbol(ref_element)

        # Shuffle the network indices
        shuffle(self.network_clust_indx)
        symb_b = ref_element
        while symb_b == ref_element:
            new_ref_element = choice(self.network_element)
            ref_indx = self.atoms_tracker.get_random_indx_of_symbol(new_ref_element)
            for indx in self.network_clust_indx:
                rand_b = self.bc.trans_matrix[ref_indx][indx]
                symb_b = self.atoms[rand_b].symbol
                if symb_b != ref_element:
                    break

        symb_a = self.atoms[rand_a].symbol
        assert symb_a == ref_element
        system_changes = [(rand_a, symb_a, symb_b),
                          (rand_b, symb_b, symb_a)]
        return system_changes

    def _get_network_statistics(self):
        """Retrieve statistics from all networks.

        :return: Network statistics
        :rtype: dict
        """
        return self.network.get_statistics()

    def _size_ok(self, stat):
        """Check if the sizes are OK.

        :param dict stat: Network statistics
        :return: True/False if the move is fine or not
        :rtype: bool
        """
        full_size = 0
        for st in stat:
            full_size += st["max_size"]

        if self.size is None:
            self.size = full_size
        return full_size == self.size

    def move_ok(self):
        """Check if the move is OK concenerning the cluster constraint.

        :return: True/False depending on if we still have only one cluster or
            not
        :rtype: bool
        """
        stat = self.cluster_stat
        n_in_clst = stat["n_atoms_in_cluster"]
        mv_ok = n_in_clst == self.initial_num_atoms_in_cluster
        mv_ok = mv_ok and stat["number_of_clusters"] == 1
        return mv_ok
        #return self.network.move_creates_new_clusters()

    def _accept(self, system_changes):
        """Accept trial move.

        :param list system_changes: Proposed changes

        :return: True/False, if True the move is accepted
        :rtype: bool
        """
        # Note that we have to call the parent's accept first,
        # as the _mc_step function assumes that an energy 
        # evaluation have been performed, prior to accepting
        # or rejecting the move
        move_accepted = Montecarlo._accept(self, system_changes)
        if self.network.move_creates_new_cluster(system_changes):
            return False

        # if not self.move_ok():
        #     return False
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
        """Get an initial site in the correct symm group.

        :return: Initial site for growing a cluster
        :rtype: int
        """
        element = self.network_element[0]
        try:
            indx = self.atoms_tracker.get_random_indx_of_symbol(element)
        except KeyError:
            raise KeyError("Could not find any seed."
                           "Insert one atom from network_elements "
                           "such that it can be used as seed for the cluster.")
        return indx

    def _indices_in_spherical_neighborhood(self, radius, root):
        """Return a list with indices in a spherical neighbor hood.

        :param float radius: Radius of the spherical neighborhood
        :param int root: Root index
        """
        indices = list(range(len(self.atoms)))
        del indices[root]
        dists = self.atoms.get_distances(root, indices, mic=True)
        return [indx for indx, d in zip(indices, dists) if d < radius]

    def init_cluster_info(self):
        """Initialize cluster info."""
        self.network.collect_statistics = True
        self.network.reset()
        self.network(None)
        stat = self.network.get_statistics()
        self.initial_num_atoms_in_cluster = stat["n_atoms_in_cluster"]

        if stat["number_of_clusters"] != 1:
            print("Cluster statistics:")
            print(stat)
            indices = []
            for atom in self.atoms:
                if atom.symbol in self.network_element:
                    indices.append(atom.index)
            cluster = self.atoms[indices]
            from ase.io import write
            fname = "invalide_cluster{}.xyz".format(self.rank)
            write(fname, cluster)
            raise RuntimeError("There are {} clusters present! "
                               "Initial structure written to {}\n"
                               "".format(stat["number_of_clusters"], fname))
        
        self.network.collect_statistics = False

    def grow_cluster(self, elements, shape="arbitrary", radius=10.0):
        """Grow a cluster of a certain size.

        :param dict elements: How many of each element that should be insreted
        :param str shape: Shape to create (either arbitrary or spherical)
        :param float radius: Radius of the spheres
        """
        from random import choice, shuffle
        valid_shapes = ["arbitrary", "sphere"]
        if shape not in valid_shapes:
            raise ValueError("shape has to be one of {}".format(valid_shapes))

        self.network.collect_statistics = True
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
        inserted_indices = [ref_site]

        if shape == "arbitrary":
            candidate_indices = list(range(len(self.atoms)))
        elif shape == "sphere":
            candidate_indices = self._indices_in_spherical_neighborhood(
                radius, ref_site
            )

        if len(candidate_indices) < len(all_elems):
            raise ValueError("The allowed indices to insert things, is "
                             "smaller than the number of elements to insert! "
                             "Num allowed: {} "
                             "Num to insert: {}"
                             "".format(len(candidate_indices), len(all_elems)))

        max_attemps = 10 * len(self.atoms)
        attempt = 0
        while all_elems and attempt < max_attemps:
            attempt += 1
            shuffle(inserted_indices)
            shuffle(self.network_clust_indx)
            element = choice(all_elems)
            found_place_to_insert = False
            for ref_site in inserted_indices:
                for net_indx in self.network_clust_indx:
                    indx = self.bc.trans_matrix[ref_site][net_indx]
                    old_symb = self.atoms[indx].symbol
                    if old_symb in unique_solute_elements:
                        continue
                    system_changes = (indx, old_symb, element)

                    valid = indx in candidate_indices
                    no_violations = self._no_constraint_violations(
                        [system_changes])

                    if no_violations and valid:
                        self.atoms.get_calculator().update_cf(system_changes)
                        all_elems.remove(element)
                        found_place_to_insert = True
                        inserted_indices.append(indx)
                        break
                if found_place_to_insert:
                    break

        if attempt == max_attemps:
            raise RuntimeError("Did not manage to create a valid cluster "
                               "in {} inertion attempts".format(max_attemps))
        self._build_atoms_list()
        self.atoms.get_calculator().clear_history()

        # Verify that the construction was fine
        at_count = self.count_atoms()
        for k, target_num in elements.items():
            if at_count[k] != target_num:
                raise ValueError("Inconsistent size!\n"
                                 "Should be: {}\n"
                                 "Contains: {}".format(elements, at_count))
        self.init_cluster_info()

        # Disable network statistics to make the program
        # run faster
        self.network.collect_statistics = False
        self.current_energy = self.atoms.get_calculator().get_energy()

    def get_atoms(self, atoms=None, prohib_elem=[]):
        """
        Return the atoms object with the clusters highlighted

        :param atoms: Atoms object, if None the attached one will be used
        :type atoms: Atoms or None
        :param list prohib_elem: List of symbols that can not be used to
            indicate atoms in the same cluster

        :return: Full atoms object and an atoms object belonging to the largest
            cluster
        :rtype: Atoms, Atoms
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

    def set_symbols(self, symbs):
        """Override parents set symbols."""
        Montecarlo.set_symbols(self, symbs)
        self.network.collect_statistics = False
        self.network([])

    def runMC(self, steps=100000, init_cluster=True, elements={}, equil=False):
        """
        Run Monte Carlo for fixed nucleus size

        :param int steps: Number of Monte Carlo steps
        :param bool init_cluster: If True initialize a cluster, If False it is
            assumed that a cluster of the correct size already exists in the
            system
        :param dict elements: Elements in the cluster
        """
        if init_cluster:
            self._check_nucleation_site_exists()
            self.grow_cluster(elements)
        self.network.collect_statistics = False

        # Call one time
        self.network([])

        if self.network.num_root_nodes() > 1:
            raise ValueError("Something went wrong during construction! "
                             "the system has more than one cluster!")
        Montecarlo.runMC(self, steps=steps, equil=equil)
