from cemc.mcmc import Montecarlo
from cemc.mcmc import NetworkObserver
import numpy as np
import random


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
        self.size = None
        super(FixedNucleusMC, self).__init__(atoms, T, **kwargs)

        self.networks = self._init_networks()
        self.bc = self.atoms._calc.BC
        self.network_clust_indx = self.find_cluster_indx()
        self._check_nucleation_site_exists()

    def _init_networks(self):
        """Initialize the network observers."""
        networks = []
        for name, elm in zip(self.network_name, self.network_element):
            network = NetworkObserver(
                calc=self.atoms._calc, cluster_name=name, element=elm)
            networks.append(network)
        return networks

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


        # self.selected_b = self.atoms_indx[symb_b].index(self.rand_b)
        # Breaks the cluster tracker but it is not needed here
        self.selected_b = 0
        symb_a = self.atoms[self.rand_a].symbol
        assert symb_a == ref_element
        system_changes = [(self.rand_a, symb_a, symb_b),
                          (self.rand_b, symb_b, symb_a)]
        return system_changes

    def _reset_networks(self):
        """Reset all the networks."""
        for network in self.networks:
            network.reset()

    def _get_network_statistics(self):
        """Retrieve statistics from all networks."""
        stat = []
        for network in self.networks:
            stat.append(network.get_statistics())
        return stat

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
        # self._reset_networks()
        # self.network(system_changes)
        # stat = self._get_network_statistics()
        # self._reset_networks()
        # if not self._size_ok(stat):
        #     return False
        return move_accepted

    def _check_nucleation_site_exists(self):
        """Check that at least one nucleation site exists."""
        for atom in self.atoms:
            if atom.symbol == self.network_element[0]:
                return
        elm = self.network_element[0]
        raise ValueError("At least one element of {}"
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
        for symb, num in elements.items():
            for _ in range(num):
                all_elems.append(symb)

        ref_site = self._get_initial_site()
        while all_elems:
            shuffle(self.network_clust_indx)

            element = choice(all_elems)
            for net_indx in self.network_clust_indx:
                indx = self.bc.trans_matrix[ref_site][net_indx]
                system_changes = [(indx, self.atoms[indx].symbol, element)]

                if self._no_constraint_violations(system_changes):
                    self.atoms.get_calculator().update_cf(system_changes)
                    all_elems.pop(indx)
                    break
            ref_site = indx
        self._build_atoms_list()
        self.atoms.get_calculator().clear_history()

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

        indices = []
        for network in self.networks():
            network.reset()
            network(None)
            indices += network.get_indices_of_largest_cluster()

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

    def runMC(self, nsteps=100000, init_cluster=True, elements={}):
        """
        Run Monte Carlo for fixed nucleus size

        :param nsteps: Number of Monte Carlo steps
        :param init_cluster: If True initialize a cluster, If False it is
            assumed that a cluster of the correct size already exists in the
            system
        """
        if init_cluster:
            self._grow_cluster(elements)
        step = 0
        while step < nsteps:
            step += 1
            self._mc_step()
        accpt_rate = float(self.num_accepted) / self.current_step
        print("Acceptance rate: {}".format(accpt_rate))
