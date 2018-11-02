import sys
from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
from cemc.mcmc.mpi_tools import num_processors, mpi_rank
import numpy as np
from itertools import product
import time
from numpy.linalg import inv


class CouldNotFindValidStateError(Exception):
    pass


class InertiaCrdInitializer(ReactionCrdInitializer):
    """Initializer for various version of principal moment of inertia.

    :param FixedNucleusMC fixed_nuc_mc: Monte Carlo object
    :param str matrix_element: Matrix element
    :param list cluster_elements: Elements in the clusters
    :param int num_matrix_atoms_surface: Number of neighboring matrix atoms
        required if a cluster atoms should be considered to be on the
        surface
    :param str traj_file: Trajectory file when the system is evolved towards
        a target value for the reaction coordinate
    :param str traj_file_clst: Trajectory file containing only the cluster
    :param int output_every: Interval in seconds for how often status
        messages should be printed
    """

    def __init__(self, fixed_nucl_mc=None, matrix_element=None,
                 cluster_elements=[], num_matrix_atoms_surface=1,
                 traj_file="full_system_insertia.traj",
                 traj_file_clst="clusters_inertial.traj",
                 output_every=10, formula="I1/I3", outlier_sensitivity=None):
        from cemc.mcmc import InertiaTensorObserver
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
        self.inert_obs = InertiaTensorObserver(atoms=fixed_nucl_mc.atoms, cluster_elements=cluster_elements)

        # Attach the inertia observer to the 
        # fixed nucleation sampler
        self.fixed_nucl_mc.attach(self.inert_obs)

        size = num_processors()
        rank = mpi_rank()
        if size > 1:
            # Rename the trajectory file writer one for each process
            fname_base = traj_file.rpartition(".")[0]
            traj_file = fname_base + str(rank) + ".traj"
            fname_base = traj_file_clst.rpartition(".")[0]
            traj_file_clst = fname_base + str(rank) + ".traj"
        self.traj_file = traj_file
        self.traj_file_clst = traj_file_clst

        if outlier_sensitivity is not None:
            if not isinstance(outlier_sensitivity, OutlierDetection):
                raise TypeError("outlier_sensitivity has to be of type "
                                "OutliderDetection or None")
        self.outlier_sensitivity = outlier_sensitivity

    def inertia_tensor(self, atoms, system_changes):
        """Calculate the inertial tensor of the cluster.

        :return: Inertia tensor
        :rtype: Numpy 3x3 matrix
        """
        if system_changes:
            self.inert_obs(system_changes)
        elif atoms is not None:
            # Perform a new calculation from scratch
            self.inert_obs.set_atoms(atoms)

        inertia = self.inert_obs.inertia

        # This class should not alter the intertia tensor
        # so we undo the changes
        if system_changes:
            self.inert_obs.undo_last()
        return inertia

    def principal_inertia(self, atoms, system_changes):
        """Calculate the inertia of the atoms in cluster elements.

        :return: Principal moment of inertia
        :rtype: numpy 1D array of length 3
        """
        eigv = np.linalg.eigvals(self.inertia_tensor(atoms, system_changes))
        return eigv

    @property
    def indices_in_cluster(self):
        """Find the indices of the atoms belonding to the cluster.

        :return: Indices of the atoms in the cluster
        :rtype: list of int
        """
        include = []
        for symb in self.cluster_elements:
            include += self.fixed_nucl_mc.atoms_tracker.tracker[symb]
        return include

    def normalized_principal_inertia(self, atoms, system_changes):
        """Principal inertia normalized by the largest component.

        :return: Normalized principal inertia
        :rtype: 1D numpy array of length 3
        """
        princ_inertia = self.principal_inertia(atoms, system_changes)
        return princ_inertia / np.max(princ_inertia)

    @property
    def dist_all_to_all(self):
        """Get distance between all atoms.

        :return: All distances between atoms in the clsuter
        :rtype: list of numpy 1D arrays
        """
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
        """Get a flattened list of all distances.

        :return: Flattened distance list
        :rtype: list of float
        """
        dists = self.dist_all_to_all
        flat_list = []
        for sublist in dists:
            flat_list += list(sublist)
        return flat_list

    def get(self, atoms, system_changes=[]):
        """Get the inertial reaction coordinate.

        :param Atoms atoms: Not used. Using the atoms object of fixed_nucl_mc.

        :return: The reaction coordinate
        :rtype: float
        """
        princ = self.principal_inertia(atoms, system_changes)
        princ = np.sort(princ)
        z_score = self.z_score_sensitivity(atoms, system_changes)

        # Make sure they are sorted in the correct order
        assert princ[0] <= princ[2]

        if self.formula == "I1/I3":
            return 1.0 - np.min(princ)/np.max(princ) + z_score
        elif self.formula == "2*I1/(I2+I3)":
            return 1.0 - 2.0 * princ[0]/(princ[1] + princ[2]) + z_score
        elif self.formula == "(I1+I2)/(2*I3)":
            return 1.0 - (princ[0] + princ[1])/(2.0*princ[2]) + z_score
        else:
            raise ValueError("Unknown formula {}".format(self.formula))

    def z_score_sensitivity(self, atoms, system_changes):
        if self.outlier_sensitivity is None:
            return 0.0
        
        if system_changes:
            self.outlier_sensitivity.update(system_changes)
        elif atoms is not None:
            self.outlier_sensitivity.calculate_from_scratch(atoms)
        return self.outlier_sensitivity.z_score_parameter()
        


    @property
    def surface_atoms(self):
        """Return a list of atoms on a surface.

        :return: Indices of the atoms on the surface
        :rtype: list of int
        """
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
        """Create an atoms object with the correct reaction coordinate.

        :param Atoms atom: Atoms object (not used, using the one attached
            to the MC object). Argument only included  traj_full = TrajectoryWriter(self.traj_file, mode="a")
        traj_clst = TrajectoryWriter(self.traj_file_clst, mode="a")parent class
            has it.
        :param float value: Target value for the react traj_full = TrajectoryWriter(self.traj_file, mode="a")
        traj_clst = TrajectoryWriter(self.traj_file_clst, mode="a")dinate
        """
        from random import choice, shuffle

        # Make sure that the observer is initialized correctly
        self.inert_obs.init_com_and_inertia()
        self.fixed_nucl_mc.network([])

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
        output_every = 15
        now = time.time()
        rank = mpi_rank()
        while attempt < max_attempts:
            if self.fixed_nucl_mc.network.num_root_nodes() > 1:
                raise RuntimeError("For some unknown reason there are "
                                   "more than one cluster!")
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

                    if self.fixed_nucl_mc.network.move_creates_new_cluster(system_changes):
                        continue

                    assert self.fixed_nucl_mc.network.num_root_nodes() == 1
                    if mc._no_constraint_violations(system_changes):
                        calc.calculate(atoms, ["energy"], system_changes)
                        found_swap_candidate = True
                        break

            if not found_swap_candidate:
                continue

            # Get bases its calculation on the atom tracker
            new_value = self.get(atoms, system_changes=system_changes)
            new_diff = abs(new_value - value)

            if time.time() - now > output_every:
                print("Rank: {} Current value: {} Target value: {}"
                      "".format(rank, new_value, value))
                sys.stdout.flush()
                now = time.time()

            if new_diff < current_diff:
                # The candidate trial moves brings the system closer to the
                # target value, so we accept this move
                current_diff = new_diff

                # We need to update the inertia observer
                self.inert_obs(system_changes)

                # Update the network
                assert self.fixed_nucl_mc.network.num_root_nodes() == 1
                self.fixed_nucl_mc.network(system_changes)
                assert self.fixed_nucl_mc.network.num_root_nodes() == 1

                # Update the symbol tracker
                self.fixed_nucl_mc._update_tracker(system_changes)
                calc.clear_history()
            else:
                calc.undo_changes()
                assert self.fixed_nucl_mc.network.num_root_nodes() == 1

            if should_increase_value and new_value > value:
                break
            elif shoud_decrease_value and new_value < value:
                break

        if attempt == max_attempts:
            raise CouldNotFindValidStateError("Did not manage to find a state "
                                              "with reaction coordinate "
                                              "{}!".format(value))


class InertiaRangeConstraint(ReactionCrdRangeConstraint):
    """Constraint to ensure that the system stays without its bounds.

    :param FixedNucleusMC fixed_nuc_mc: Monte Carlo object
    :param list range: Upper and lower bound of the reaction coordinate
    :param InertiaCrdInitializer inertia_init: Initializer
    :param bool verbose: If True print messages every 10 sec
        if the constraint is violated
    """

    def __init__(self, fixed_nuc_mc=None, range=[0.0, 1.0], inertia_init=None,
                 verbose=False):
        super(InertiaRangeConstraint, self).__init__()
        self.update_range(range)
        self.mc = fixed_nuc_mc
        self._inertia_init = inertia_init
        self.last_print = time.time()
        self.verbose = verbose

    def get_new_value(self, system_changes):
        """Get new value for reaction coordinate.

        :param list system_changes: List with the proposed changes

        :return: Reaction coordate after the change
        :rtype: float
        """

        # Get the new value of the observer
        new_val = self._inertia_init.get(None, system_changes=system_changes)
        return new_val

    def __call__(self, system_changes):
        """Check the system is in a valid state after the changes.

        :param list system_changes: Proposed changes

        :return: True/False, if True the system is still within the bounds
        :rtype: bool
        """
        new_val = self.get_new_value(system_changes)
        ok = (new_val >= self.range[0] and new_val < self.range[1])

        if not ok and self.verbose:
            # The evaluation of this constraint can be time consuming
            # so let the user know at regular intervals
            rank = mpi_rank()
            if time.time() - self.last_print > 10:
                print("Move violates constraint on rank {}".format(rank))
                self.last_print = time.time()
        return ok

class OutlierDetection(object):
    """
    Class that calculates 
    """
    def __init__(self, inert_obs=None, roots=[0.0, 1.0], amp=1.0):
        self.insert_obs = inert_obs
        self.indx = 0
        self.current_z_score = 0.0
        self.roots = np.array(roots)
        self.amp = amp

    def z_score_parameter(self, z_score=None):
        """Return the z-score parameter."""
        if z_score is None:
            z_score = self.current_z_score
        return np.abs(np.prod(z_score - self.roots)*self.amp)

    def get_z_score(self, vec):
        """Return the z score."""
        C_inv = inv(self.insert_obs.inertia)
        return vec.T.dot(C_inv.dot(vec))

    def get_new(self, system_changes):
        """Update the furthest index."""
        atoms = self.insert_obs.atoms
        largest_z = self.current_z_score
        indx = self.indx
        for change in system_changes:
            if change[2] in self.insert_obs.cluster_elements:
                diff = atoms[change[0]].position - self.insert_obs.com
                z_score = self.get_z_score(diff)
                if z_score > largest_z:
                    largest_z = z_score
                    indx = change[0]
        return indx, largest_z

    def update(self, indx):
        """Update the maximum z score."""
        self.indx = indx
        com = self.insert_obs.com
        pos = self.insert_obs[indx].position
        vec = pos - com
        self.current_z_score = self.get_z_score(vec)

        # Make sure that we actually have tracked a 
        # cluster element
        atoms = self.insert_obs.atoms
        symbs = self.insert_obs.cluster_elements
        if atoms[indx] not in symbs:
            self.calculate_from_scratch(atoms)

    def calculate_from_scratch(self, atoms):
        """Calculate the z score from scratch."""
        symbs = self.insert_obs.cluster_elements
        indices = [atom.index for atom in atoms 
                   if atom.symbol in symbs]
        pos = atom.get_positions()[indices, :]
        pos -= self.insert_obs.com
        z_scores = self.get_z_score(pos.T)
        max_indx = np.argmax(z_scores)
        self.current_z_score = z_scores[max_indx]
        self.indx = indices[max_indx]
                

    