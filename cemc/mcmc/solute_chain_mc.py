from cemc.mcmc import Montecarlo
from cemc.mcmc import MCObserver
from cemc.mcmc import FixEdgeLayers
from random import choice
import numpy as np
from copy import deepcopy
from scipy.spatial import cKDTree as KDTree


class SoluteChainMC(Montecarlo):
    def __init__(self, atoms, T, cluster_elements=[], cluster_names=[], 
                 move_weight=[("solute-solute", 0.5), ("solute-matrix", 0.5)],
                 edge_layer=2):
        Montecarlo.__init__(self, atoms, T)
        if not cluster_names:
            raise ValueError("No cluster names given!")
        self.cluster_elements = cluster_elements
        self.matrix_elements = list(set(self.symbols) - set(self.cluster_elements))
        self.max_attempts = 10000

        # Add edge layer constraint
        thickness = self._get_nn_distance(edge_layer)
        self.edge_layer = FixEdgeLayers(thickness=thickness, atoms=self.atoms)
        self.add_constraint(self.edge_layer)
        
        tm = self.atoms.get_calculator().BC.trans_matrix
        self.connectivity = SoluteConnectivity(self.atoms, tm, cluster_elements, cluster_names)

        # Attach the connectivity observer
        self.attach(self.connectivity)
        self.move_weigth_names = [mv[0] for mv in move_weight]
        self.move_dist = np.cumsum([mv[1] for mv in move_weight])
        self.move_dist /= self.move_dist[-1]

    def build_chain(self, num_solutes={}):
        """Build a chain of solute atoms."""
        self.connectivity.build_chain(num_solutes, constraints=self.constraints)
        self._build_atoms_list()
        self.current_energy = self.atoms.get_calculator().get_energy()

    def _get_nn_distance(self, nn):
        """Return the neighbour distance."""
        dists = self.atoms.get_distances(0, range(1, len(self.atoms)))
        dists.sort()
        return dists[nn]

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
        randnum = np.random.rand()
        if self.connectivity.has_flexible_sites and randnum < 0.5:
            rand1 = self.connectivity.get_random_flexible_index()
            symb1 = self.atoms[rand1].symbol
            symb2 = symb1
            while symb2 in self.cluster_elements:
                rand2 = self.connectivity.get_random_neighbour_index(rand1)
                symb2 = self.atoms[rand2].symbol
        else:
            rand1 = self.connectivity.get_random_single_connected()
            symb1 = self.atoms[rand1].symbol
            symb2 = symb1
            while symb2 in self.cluster_elements:
                rand_sol = self.atoms_tracker.get_random_indx_of_symbol(choice(self.cluster_elements))
                rand2 = self.connectivity.get_random_neighbour_index(rand_sol)
                symb2 = self.atoms[rand2].symbol
        
        assert symb1 in self.cluster_elements
        assert symb2 in self.matrix_elements

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
        self.single_connected_points = self.get_single_connected_points()

        self.tree = KDTree(self.atoms.get_positions())
        self.max_dist = self._get_max_distance_in_cluster()

    def _get_max_distance_in_cluster(self):
        """Calculate the maximum distance in the cluster."""
        dists = self.atoms.get_distances(0, self.cluster_indices)
        return max(dists)


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

    def violate_constraints(self, change, constraints):
        for cnst in constraints:
            if not cnst(change):
                return True
        return False

    def build_chain(self, num_solutes={}, constraints=[]):
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
                    indx = neighbor[neighbor_cnt]
                    neighbor_cnt += 1

                if neighbor_cnt == len(neighbor):
                    root_indx_cnt += 1
                    root = inserted[root_indx_cnt]
                    neighbor = self._neighbour_indices(root)
                    neighbor_cnt = 0
                    continue

                # Proposed change
                change = [(indx, self.atoms[indx].symbol, symb)]
                if self.violate_constraints(change, constraints):
                    continue
                symbols[indx] = symb
                self.connectivity[indx] = [prev_inserted]
                self.connectivity[prev_inserted].append(indx)
                prev_inserted = indx
                inserted.append(indx)
                num_inserted += 1

        self.atoms.get_calculator().set_symbols(symbols)
        self.flexible_indices = self.get_flexible_indices()
        self.single_connected_points = self.get_single_connected_points()
        assert self.connectivity_is_consistent()
        assert all([self.atoms[i].symbol in self.cluster_elements for i in self.flexible_indices])

    def get_flexible_indices(self):
        """Return a list of indices that supports moves outside chain."""
        flex_index = []
        for atom in self.atoms:
            if atom.symbol not in self.cluster_elements:
                continue

            if self.is_flexible(atom.index):
                flex_index.append(atom.index)
        return flex_index

    def __call__(self, system_changes):
        """Update connectivity if a swap move."""

        if system_changes[0][1] == system_changes[0][2]:
            # Move was rejected (no changes introduced)
            return
        elif system_changes[0][1] in self.cluster_elements and system_changes[0][2] in self.cluster_elements:
            # Both elements are solute elements. Swapping these have 
            # no impact on the connectivity
            return

        if system_changes[0][1] in self.cluster_elements:
            old_indx = system_changes[0][0]
            new_indx = system_changes[1][0]
        else:
            old_indx = system_changes[1][0]
            new_indx = system_changes[0][0]

        self.connectivity[new_indx] = [] 
        temp_connect = deepcopy(self.connectivity[old_indx])

        self.connectivity[old_indx] = []
        
        for indx in temp_connect:
            self.connectivity[indx].remove(old_indx)

            if new_indx in self._neighbour_indices(indx):
                self.connectivity[indx].append(new_indx)
                self.connectivity[new_indx].append(indx)

            if len(self.connectivity[indx]) == 1 and indx not in self.single_connected_points:
                self.single_connected_points.append(indx)

        if old_indx in self.single_connected_points:
            self.single_connected_points.remove(old_indx)

        if len(self.connectivity[new_indx]) == 0:
            # The new index is not connected to any of the old.
            # We need to perform a search to find a connection
            # point
            candidates = self.tree.query_ball_point(self.atoms[new_indx].position, self.max_dist)

            # There should never be no neighbours.
            # If that is the case something is wrong
            assert candidates

            for cand in candidates:
                if cand == new_indx:
                    continue
                if self.atoms[cand].symbol in self.cluster_elements:
                    self.connectivity[new_indx].append(cand)
                    self.connectivity[cand].append(new_indx)
                    break

        if len(self.connectivity[new_indx]) == 1:
            self.single_connected_points.append(new_indx)

        if old_indx in self.flexible_indices:
            self.flexible_indices.remove(old_indx)

        # Check if we have new flexible indices
        indices = self.connectivity[new_indx] + [new_indx]

        for indx in indices:
            if self.is_flexible(indx) and indx not in self.flexible_indices:
                self.flexible_indices.append(indx)

        # There are always two end points, so there has
        # to be at least 2 single connected points
        assert len(self.single_connected_points) >= 2

    def connectivity_is_consistent(self):
        """Check that connectivity is consistent."""
        for i, con in enumerate(self.connectivity):
            for j in con:
                if i not in self.connectivity[j]:
                    return False

        # Check that all elements in the connectivity lists
        # are solute atoms
        for con in self.connectivity:
            for item in con:
                if self.atoms[item].symbol not in self.cluster_elements:
                    return False

        # Check that the chain has exactly two end points
        if self.num_end_points != 2:
            return False
        return True

    @property
    def num_end_points(self):
        num_end = 0
        for con in self.connectivity:
            if len(con) == 1:
                num_end += 1
        return num_end

    def is_flexible(self, root):
        return len(self._matrix_indices_preserving_connectivity(root)) > 0
    
    def _indices_preserving_connectivity(self, root):
        """Return a list of indices preserving the connectivity."""
        indx = self.connectivity[root][0]
        connect_preserving_indx = set(self._neighbour_indices(indx))
        for i in range(1, len(self.connectivity[root])):
            indx = self.connectivity[root][i]
            connect_preserving_indx = connect_preserving_indx.intersection(
                set(self._neighbour_indices(indx))
            )
        return list(connect_preserving_indx)

    def _matrix_indices_preserving_connectivity(self, root):
        """Return a list of indices preserving the connectivity."""
        indx = self.connectivity[root][0]
        connect_preserving_indx = set(self._neighbour_matrix_indices(indx))
        for i in range(1, len(self.connectivity[root])):
            indx = self.connectivity[root][i]
            connect_preserving_indx = connect_preserving_indx.intersection(
                set(self._neighbour_matrix_indices(indx))
            )
        return list(connect_preserving_indx)

    def _index_at_center(self):
        """Return the index of the atoms closest to the center."""
        diag = self.atoms.get_cell().T.dot([0.5, 0.5, 0.5])
        pos = self.atoms.get_positions() - diag
        lengths = np.sum(pos**2, axis=1)
        return np.argmin(lengths)

    def _neighbour_indices(self, root):
        return [self.tm[root][x] for x in self.cluster_indices]

    def _neighbour_solute_indices(self, root):
        neighbor = self._neighbour_indices(root)
        return [index for index in neighbor 
                if self.atoms[index].symbol in self.cluster_elements and index != root]

    def _neighbour_matrix_indices(self, root):
        neighbor = self._neighbour_indices(root)
        return [index for index in neighbor if self.atoms[index].symbol not in self.cluster_elements]

    def get_random_neighbour_index(self, root):
        """Return a random neighbour index."""
        return choice(self._neighbour_indices(root))

    def has_matrix_element_neighbours(self, root):
        """Check if the current element has matrix element neighbours."""
        return any([self.atoms[i].symbol not in self.cluster_elements 
            for i in self._neighbour_indices(root)])

    def get_random_single_connected(self):
        return choice(self.single_connected_points)

    def get_random_flexible_index(self):
        """Return a random index that can support"""
        return choice(self.flexible_indices)

    def get_single_connected_points(self):
        single_con = []
        for i, con in enumerate(self.connectivity):
            if len(con) == 1:
                single_con.append(i)
        return single_con

    @property
    def has_flexible_sites(self):
        return len(self.flexible_indices) > 0

    @property
    def has_single_connected_points(self):
        return len(self.single_connected_points) > 0