from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
import numpy as np
from itertools import combinations_with_replacement


class CouldNotFindValidStateError(Exception):
    pass


class InertiaCrdInitializer(ReactionCrdInitializer):
    def __init__(self, fixed_nucl_mc=None, matrix_element=None,
                 cluster_elements=[], num_matrix_atoms_surface=1):
        if matrix_element in cluster_elements:
            raise ValueError("InertiaCrdInitializer works only when "
                             "the matrix element is not present in the "
                             "clustering element!")
        self.matrix_element = matrix_element
        self.cluster_elements = cluster_elements
        self.fixed_nucl_mc = fixed_nucl_mc
        self.num_matrix_atoms_surface = num_matrix_atoms_surface

    @property
    def inertia_tensor(self):
        include = self.indices_in_cluster
        cluster = self.fixed_nucl_mc.atoms[include]
        cluster = InertiaCrdInitializer.center_atoms(cluster)
        pos = cluster.get_positions()
        com = np.sum(pos, axis=0) / pos.shape[0]
        assert len(com) == 3

        inertia_tensor = np.zeros((3, 3))
        pos -= com
        for comb in combinations_with_replacement(list(range(3)), 2):
            i1 = comb[0]
            i2 = comb[1]
            inertia_tensor[i1, i2] = np.sum(pos[:, i1] * pos[:, i2])
        inertia_tensor += inertia_tensor.T
        inertia_tensor *= 0.5
        return inertia_tensor

    @staticmethod
    def center_atoms(atoms):
        """Center the atoms in the cell."""
        cell = atoms.get_cell()
        diag = 0.5 * (cell[0, :] + cell[1, :] + cell[2, :])
        indx = list(range(1, len(atoms)))
        com = np.sum(atoms.get_distances(0, indx, mic=True, vector=True),
                     axis=0)/len(atoms)
        com += atoms[0].position
        atoms.translate(diag - com)
        atoms.wrap()
        return atoms

    @property
    def principal_inertia(self):
        """Calculate the inertia of the atoms in cluster elements"""
        eigv = np.linalg.eigvals(self.inertia_tensor)
        return eigv

    @property
    def indices_in_cluster(self):
        include = []
        for symb in self.cluster_elements:
            include += self.fixed_nucl_mc.atoms_indx[symb]
        return include

    @property
    def normalized_principal_inertia(self):
        """Principal inertia normalized by the largest component."""
        princ_inertia = self.principal_inertia
        return princ_inertia / np.max(princ_inertia)

    def view_cluster(self):
        """View the cluster used for for inertial calculation."""
        from ase.visualize import view
        include = self.indices_in_cluster
        cluster = self.fixed_nucl_mc.atoms[include]
        cluster = InertiaCrdInitializer.center_atoms(cluster)
        view(cluster)

    def get(self, atoms):
        """Get the inertial reaction coordinate.

        :param atoms: Not used. Using the atoms object of fixed_nucl_mc.
        """
        norm_inert = self.normalized_principal_inertia
        norm_inert = np.sort(norm_inert)
        return 1.0 - 2.0 * norm_inert[0]/(norm_inert[1] + norm_inert[2])

    @property
    def surface_atoms(self):
        """Return a list of atoms on a surface."""
        indx = np.array(self.indices_in_cluster)
        neighbors = self.fixed_nucl_mc.network_clust_indx
        num_matrix_atoms = np.zeros(len(indx))
        for j, i in enumerate(indx):
            for t in neighbors:
                tr_indx = self.fixed_nucl_mc.get_translated_indx(i, t)
                symb = self.fixed_nucl_mc.atoms[tr_indx].symbol
                if symb == self.matrix_element:
                    num_matrix_atoms[j] += 1
        return indx[num_matrix_atoms >= self.num_matrix_atoms_surface]

    def set(self, atoms, value):
        """Create an atoms object with the correct reaction coordinate."""
        from random import choice, shuffle
        max_attempts = 1000 * len(self.fixed_nucl_mc.atoms)
        attempt = 0
        neighbors = self.fixed_nucl_mc.network_clust_indx
        atoms = self.fixed_nucl_mc.atoms
        calc = atoms.get_calculator()
        current_value = self.get(atoms)
        current_diff = abs(value - current_value)

        should_increase_value = current_diff < value
        shoud_decrease_value = not should_increase_value
        mc = self.fixed_nucl_mc
        while attempt < max_attempts:
            attempt += 1
            surf_atoms = self.surface_atoms()
            rand_surf_atom = choice(surf_atoms)
            rand_surf_atom2 = choice(surf_atoms)
            shuffle(neighbors)
            found_swap_candidate = False
            for indx in neighbors:
                t_indx = mc.get_translated_indx(rand_surf_atom2, indx)
                symb = mc.atoms[t_indx].symbol
                if symb == self.matrix_element:
                    old_symb = mc.atoms[rand_surf_atom].symbol
                    ch1 = (rand_surf_atom, old_symb, symb)
                    ch2 = (t_indx, symb, old_symb)
                    system_changes = [ch1, ch2]

                    if mc._no_constraint_violations(system_changes):
                        calc.calculate(atoms, ["energy"], system_changes)
                        found_swap_candidate = True
                        break
                else:
                    continue
            if not found_swap_candidate:
                continue

            new_value = self.get(atoms)
            new_diff = abs(new_value - value)

            if new_diff < current_diff:
                # The candidate trial moves brings the system closer to the
                # target value, so we accept this move
                current_diff = new_diff
                calc.clear_history()
                self.mc._update_tracker(system_changes)
            else:
                calc.undo_changes()

            if should_increase_value and new_value > value:
                break
            elif shoud_decrease_value and new_value < value:
                break

        if attempt == max_attempts:
            raise CouldNotFindValidStateError("Did not manage to find a state "
                                              "with reaction coordinate "
                                              "{}!".format(value))


class InertiaRangeConstraint(ReactionCrdRangeConstraint):
    def __init__(self, fixed_nuc_mc=None, range=[0.0, 1.0], inertia_init=None):
        super(InertiaCrdInitializer, self).__init__(self)
        self.update_range(range)
        self.mc = fixed_nuc_mc
        self._inertia_init = inertia_init

    def get_new_value(self, system_changes):
        # This is not the most efficient way to to things, but it should work
        orig_atoms = self.mc.atoms.copy()

        for change in system_changes:
            orig_symb = self.mc.atoms[change[0]].symbol
            assert orig_symb == change[1]
            self.mc.atoms[change[0]].symbol = change[2]
        new_val = self._insertia_init.get(None)

        # Reset the atoms object
        self.mc.atoms = orig_atoms
        return new_val

    def __call__(self, system_changes):
        new_val = self.get_new_value(system_changes)
        return new_val >= self.range[0] and new_val < self.range[1]
