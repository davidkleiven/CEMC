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
        self.size = kwargs.pop("size")
        self.network_element = kwargs.pop("network_element")
        self.network_name = kwargs.pop("network_name")

        super(FixedNucleusMC, self).__init__(atoms, T, **kwargs)

        self.networks = self._init_networks()
        self.bc = self.atoms._calc.BC
        self.network_clust_indx = self.find_cluster_indx()

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
            new_indx = []
            for clst in clusters["indices"]:
                network_indx.append(clst[0])
            network_indx += new_indx
        return np.unique(network_indx)

    def _get_trial_move(self):
        """Generate a trial move."""
        Na = len(self.atoms_indx[self.network_element])
        symb_b = self.network_element
        # max_rand = len(self.network_clust_indx)
        self.selected_a = np.random.randint(low=0, high=Na)
        self.rand_a = self.atoms_indx[self.network_element][self.selected_a]

        # Shuffle the network indices
        random.shuffle(self.network_clust_indx)

        while symb_b == self.network_element:
            temp_rand_num_b = np.random.randint(low=0, high=Na)
            temp_b = self.atoms_indx[self.network_element][temp_rand_num_b]
            for clust_indx in self.network_clust_indx:
                self.rand_b = self.bc.trans_matrix[temp_b][clust_indx[0]]
                symb_b = self.atoms[self.rand_b].symbol
                if symb_b != self.network_element:
                    break

        # self.selected_b = self.atoms_indx[symb_b].index(self.rand_b)
        # Breaks the cluster tracker but it is not needed here
        self.selected_b = 0
        symb_a = self.atoms[self.rand_a].symbol
        symb_b = self.atoms[self.rand_b].symbol
        system_changes = [(self.rand_a, symb_a, symb_b),
                          (self.rand_b, symb_b, symb_a)]
        return system_changes

    def _accept(self, system_changes):
        """Accept trial move."""
        move_accepted = Montecarlo._accept(self, system_changes)
        self.network.reset()
        self.network(system_changes)
        stat = self.network.get_statistics()
        self.network.reset()
        if stat["max_size"] != self.size:
            return False
        return move_accepted

    def get_atoms(self, atoms=None, prohib_elem=[]):
        """
        Return the atoms object with the clusters highlighted
        """
        # atoms = self.network.get_atoms_with_largest_cluster()
        if atoms is None:
            atoms = self.atoms
        else:
            symbs = [atom.symbol for atom in atoms]
            self.atoms._calc.set_symbols(symbs)
            atoms = self.atoms

        self.network.reset()
        self.network(None)
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

    def run(self, nsteps=100000, init_cluster=True):
        """
        Run Monte Carlo for fixed nucleus size

        :param nsteps: Number of Monte Carlo steps
        :param init_cluster: If True initialize a cluster, If False it is
            assumed that a cluster of the correct size already exists in the
            system
        """
        if init_cluster:
            self.network.grow_cluster(self.size)
            self._build_atoms_list()
            self.atoms._calc.clear_history()
        step = 0
        while step < nsteps:
            step += 1
            self._mc_step()
        accpt_rate = float(self.num_accepted) / self.current_step
        print("Acceptance rate: {}".format(accpt_rate))
