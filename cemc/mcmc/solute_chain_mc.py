from cemc.mcmc import Montecarlo
from cemc.mcmc import MCObserver
from random import choice
import numpy as np


class SoluteChainMC(Montecarlo):
    def __init__(self, atoms, T, cluster_elements=[], cluster_names=[], 
                 move_weight=[("solute-solute", 0.5), ("solute-matrix", 0.5)]):
        Montecarlo.__init__(self, atoms, T)
        self.cluster_elements = set(cluster_elements)
        self.matrix_elements = set(self.symbols) - self.cluster_elements
        self.max_attempts = 10000
        
        tm = self.atoms.get_calculator().BC.trans_matrix
        self.connectivity = SoluteConnectivity(self.atoms, tm, cluster_elements, cluster_names)

        # Attach the connectivity observer
        self.attach(self.connectivity)
        self.move_weigth_names = [mv[0] for mv in move_weight]
        self.move_dist = np.cumsum([mv[1] for mv in move_weight])
        self.move_dist /= self.move_dist[-1]

    def _swap_two_solute_atoms(self):
        """Swap two solute atoms."""
        symb1 = choice(self.cluster_elements)
        symb2 = symb1
        while symb2 == symb1:
            symb2 = choice(self.cluster_elements)

        # Find an index
        rand1 = self.atoms_tracker.get_random_indx_of_symbol(symb1)
        rand2 = self.atoms_tracker.get_random_indx_of_symbol(symb2)
        syst_change = [(rand1, symb1, symb2), (rand2, symb2, symb1)]
        return syst_change

    def _swap_solute_matrix_atoms(self):
        """Swap a solute atom with matrix atom."""
        rand1 = self.connectivity.get_random_flexible_index()
        symb1 = self.atoms[rand1].symbol
        rand2 = self.connectivity.get_random_neighbour_index(rand1)
        symb2 = self.atoms[rand2].symbol
        syst_change = [(rand1, symb1, symb2), (rand2, symb2, symb1)]
        return syst_change

    def get_move_type(self):
        """Return a random move weight."""
        indx =  np.argmin(np.random.rand() > self.move_dist)
        return self.move_weigth_names[indx]

    def _get_trial_move(self):
        """Return a trial move."""
        move = self.get_move_type()
        if move == "solute-solute":
            return self._swap_two_solute_atoms()
        elif move == "solute-matrix":
            return self._swap_solute_matrix_atoms()



class SoluteConnectivity(MCObserver):
    def __init__(self, atoms, trans_matrix, cluster_elements, cluster_names):
        self.atoms = atoms
        self.tm = trans_matrix
        self.connectivity = [[] for _ in range(len(self.atoms))]
        self.cluster_elements = cluster_elements
        self.cluster_indices = self._get_cluster_indices(cluster_names)
        self.flexible_indices = self.get_flexible_indices()

    def _get_cluster_indices(self, names):
        """Extract the cluster indices."""
        indices = []
        bc = self.atoms.get_calculator().BC
        for name in names:
            for info in bc.cluster_info_by_name(name):
                indices += self._indices_in_info(info)
        return list(set(indices))

    def _indices_in_info(self, info):
        """Extract all the indices in the info dict."""
        indices = []
        for indx in info["indices"]:
            indices.append(indx[0])
        return indices

    def build_chain(self, num_solutes={}):
        """Construct a chain with solute atoms

        :param dict num_solutes: Dictionary saying how many of
            each solute atom to insert
        """
        start_indx = self._index_at_center()
        symbols = [atom.symbol for atom in self.atoms]

        prev_inserted = -1
        inserted = []
        root_indx_cnt = 0
        neighbor_cnt = 0

        self.connectivity[start_indx] = []
        prev_inserted = start_indx
        symb = list(num_solutes.keys())[0]
        num_solutes[symb] -= 1
        symbols[start_indx] = symb
        neighbor = self._neighbour_indices(start_indx)
        inserted.append(start_indx)

        for symb, num in num_solutes.items():
            num_inserted = 0
            while num_inserted < num:
                indx = neighbor[neighbor_cnt]
                while symbols[indx] in self.cluster_elements and neighbor_cnt < len(neighbor):
                    neighbor_cnt += 1
                    indx = neighbor[neighbor_cnt]

                if neighbor_cnt == len(neighbor):
                    root_indx_cnt += 1
                    root = inserted[root_indx_cnt]
                    neighbor = self._neighbour_indices(root)
                    neighbor_cnt = 0
                    continue

                symbols[indx] = symb
                self.connectivity[indx] = [prev_inserted]
                self.connectivity[prev_inserted].append(indx)
                prev_inserted = indx
                inserted.append(indx)
                num_inserted += 1

        self.atoms.get_calculator().set_symbols(symbols)

    def get_flexible_indices(self):
        """Return a list of indices that supports moves outside chain."""
        flex_index = []
        for atom in self.atoms:
            if atom.symbol not in self.cluster_elements:
                continue

            indx = self._indices_preserving_connectivity(atom.indx)
            if len(indx) > 0:
                flex_index.append(atom.index)
        return flex_index

    def __call__(self, system_changes):
        """Update connectivity if a swap move."""

        if system_changes[0][1] == system_changes[0][2]:
            # Move was rejected (no changes introduced)
            return
            
        if system_changes[0][1] in self.cluster_elements:
            old_indx = system_changes[0][0]
            new_indx = system_changes[1][0]
        else:
            old_indx = system_changes[1][0]
            new_indx = system_changes[0][0]

        self.connectivity[new_indx] = self.connectivity[old_indx]
        self.connectivity[old_indx] = []

        # Update the connectivity of the indices connected
        for indx in self.connectivity[new_indx]:
            pos = self.connectivity[indx].index(old_indx)
            self.connectivity[indx][pos] = new_indx

        self.flexible_indices.remove(old_indx)

        # Check if we have new flexible indices
        indices = self.connectivity[new_indx] + [new_indx]
        for indx in indices:
            if self.is_flexible(indx):
                self.flexible_indices.append(indx)

    def is_flexible(self, root):
        return len(self._indices_preserving_connectivity) > 0
    
    def _indices_preserving_connectivity(self, root):
        """Return a list of indices preserving the connectivity."""
        connect_preserving_indx = set(self._neighbour_indices(root))
        for indx in self.connectivity[root]:
            connect_preserving_indx = connect_preserving_indx.intersection(
                set(self._neighbour_indices(indx))
            )
        return connect_preserving_indx

    def _index_at_center(self):
        """Return the index of the atoms closest to the center."""
        diag = self.atoms.get_cell().T.dot([0.5, 0.5, 0.5])
        pos = self.atoms.get_positions() - diag
        lengths = np.sum(pos**2, axis=1)
        return np.argmin(lengths)

    def _neighbour_indices(self, root):
        return [self.tm[root][x] for x in self.cluster_indices]

    def get_random_neighbour_index(self, root):
        """Return a random neighbour index."""
        return choice(self._neighbour_indices(root))

    def get_random_flexible_index(self):
        """Return a random index that can support"""
        return choice(self.flexible_indices)