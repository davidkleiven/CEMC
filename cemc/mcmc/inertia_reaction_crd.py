from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
import numpy as np
from itertools import product
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD


class CouldNotFindValidStateError(Exception):
    pass


class InertiaCrdInitializer(ReactionCrdInitializer):
    def __init__(self, fixed_nucl_mc=None, matrix_element=None,
                 cluster_elements=[], num_matrix_atoms_surface=1,
                 traj_file="full_system_insertia.traj",
                 traj_file_clst="clusters_inertial.traj",
                 output_every=10, formula="I1/I3"):
        if matrix_element in cluster_elements:
            raise ValueError("InertiaCrdInitializer works only when "
                             "the matrix element is not present in the "
                             "clustering element!")
        allowed_types = ["I1/I3", "2*I1/(I2+I3)", "(I1+I2)/(2*I3)"]
        if formula not in allowed_types:
            raise ValueError("formula has to be one of {}"
                             "".format(allowed_types))
        self.formula = formula
        self.matrix_element = matrix_element
        self.cluster_elements = cluster_elements
        self.fixed_nucl_mc = fixed_nucl_mc
        self.num_matrix_atoms_surface = num_matrix_atoms_surface
        self.output_every = output_every

        size = comm.Get_size()
        rank = comm.Get_rank()
        if size > 1:
            # Rename the trajectory file writer one for each process
            fname_base = traj_file.rpartition(".")[0]
            traj_file = fname_base + str(rank) + ".traj"
            fname_base = traj_file_clst.rpartition(".")[0]
            traj_file_clst = fname_base + str(rank) + ".traj"
        self.traj_file = traj_file
        self.traj_file_clst = traj_file_clst

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
        for comb in product(list(range(3)), repeat=2):
            i1 = comb[0]
            i2 = comb[1]
            inertia_tensor[i1, i2] = np.sum(pos[:, i1] * pos[:, i2])

        if not np.allclose(inertia_tensor, inertia_tensor.T):
            msg = "Inertia tensor is not symmetric!\n"
            msg += "Inertia tensor: {}\n".format(inertia_tensor)
            raise RuntimeError(msg)
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
            include += self.fixed_nucl_mc.atoms_tracker.tracker[symb]
        return include

    @property
    def normalized_principal_inertia(self):
        """Principal inertia normalized by the largest component."""
        princ_inertia = self.principal_inertia
        return princ_inertia / np.max(princ_inertia)

    def get_cluster(self):
        """Get atoms object with only the cluster."""
        include = self.indices_in_cluster
        cluster = self.fixed_nucl_mc.atoms[include]
        cluster = InertiaCrdInitializer.center_atoms(cluster)
        return cluster

    def view_cluster(self):
        """View the cluster used for for inertial calculation."""
        from ase.visualize import view
        view(self.get_cluster())

    @property
    def dist_all_to_all(self):
        """Get distance between all atoms."""
        indx = self.indices_in_cluster
        cluster = self.fixed_nucl_mc.atoms[indx]
        all_distances = []
        for indx in range(len(cluster)):
            all_indx = list(range(len(cluster)))
            del all_indx[indx]
            dists = cluster.get_distances(indx, all_indx, mic=True)
            all_distances.append(dists)
        return all_distances

    @property
    def dist_all_to_all_flattened(self):
        """Get a flattened list of all distances."""
        dists = self.dist_all_to_all
        flat_list = []
        for sublist in dists:
            flat_list += list(sublist)
        return flat_list

    def get(self, atoms):
        """Get the inertial reaction coordinate.

        :param atoms: Not used. Using the atoms object of fixed_nucl_mc.
        """
        princ = self.principal_inertia

        if self.formula == "I1/I3":
            return 1.0 - np.min(princ)/np.max(princ)
        elif self.formula == "2*I1/(I2+I3)":
            return 1.0 - 2.0 * princ[0]/(princ[1] + princ[2])
        elif self.formula == "(I1+I2)/(2*I3)":
            return 1.0 - (princ[0] + princ[1])/(2.0*princ[2])
        else:
            raise ValueError("Unknown formula {}".format(self.formula))

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

    def log(self, msg):
        print(msg)

    def set(self, atoms, value):
        """Create an atoms object with the correct reaction coordinate."""
        from random import choice, shuffle
        from ase.io.trajectory import TrajectoryWriter
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
        mc.selected_a = None
        mc.selected_b = None
        mc.rand_a = None
        mc.rand_b = None
        output_every = 15
        now = time.time()
        traj_full = TrajectoryWriter(self.traj_file, mode="a")
        traj_clst = TrajectoryWriter(self.traj_file_clst, mode="a")
        while attempt < max_attempts:
            attempt += 1
            surf_atoms = self.surface_atoms
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

            # Get bases its calculation on the atom tracker
            mc._update_tracker(system_changes)
            new_value = self.get(atoms)
            new_diff = abs(new_value - value)

            if time.time() - now > output_every:
                self.log("Current value: {} Target value: {}"
                         "".format(new_value, value))
                now = time.time()
                atoms = mc.atoms
                traj_full.write(atoms)
                cluster = self.get_cluster()
                traj_clst.write(cluster)
                atoms.set_calculator(calc)

            if new_diff < current_diff and mc.move_ok():
                # The candidate trial moves brings the system closer to the
                # target value, so we accept this move
                current_diff = new_diff
                calc.clear_history()
            else:
                calc.undo_changes()

                # If rejected we have to revert the tracker
                opposite_change = []
                for change in system_changes:
                    new_change = (change[0], change[2], change[1])
                    opposite_change.append(new_change)
                mc._update_tracker(opposite_change)

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
        super(InertiaRangeConstraint, self).__init__()
        self.update_range(range)
        self.mc = fixed_nuc_mc
        self._inertia_init = inertia_init

    def get_new_value(self, system_changes):
        self.mc.selected_a = None
        self.mc.selected_b = None

        # This function can be called on various states of a MC run
        # Sometimes the proposed move have been performed by another
        # operation, in that case we don't alter the atoms object
        # In other cases it has not been performed. In that case
        # introduce the change, and reverse it at the end
        orig_symb = self.mc.atoms[system_changes[0][0]].symbol
        move_has_been_performed = (orig_symb != system_changes[0][1])

        if not move_has_been_performed:
            # Introduce the changes to the atoms object
            for change in system_changes:
                orig_symb = self.mc.atoms[change[0]].symbol
                assert orig_symb == change[1]
                self.mc.atoms[change[0]].symbol = change[2]
            self.mc._update_tracker(system_changes)
        else:
            # Just make sure that nothing wrong happened
            assert orig_symb == system_changes[0][2]

        # Get the new value of the observer
        new_val = self._inertia_init.get(None)

        if not move_has_been_performed:
            # Construct the opposite change
            opposite_change = []
            for change in system_changes:
                new_change = (change[0], change[2], change[1])
                opposite_change.append(new_change)

            # Undo the changes
            for change in opposite_change:
                assert self.mc.atoms[change[0]].symbol == change[1]
                self.mc.atoms[change[0]].symbol = change[2]
            self.mc._update_tracker(opposite_change)
        return new_val

    def __call__(self, system_changes):
        new_val = self.get_new_value(system_changes)
        return new_val >= self.range[0] and new_val < self.range[1]
