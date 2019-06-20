import sys
from cemc.mcmc import ReactionCrdInitializer, ReactionCrdRangeConstraint
import numpy as np
from itertools import product
import time
from numpy.linalg import inv


class CouldNotFindValidStateError(Exception):
    pass


class CovarianceCrdInitializer(ReactionCrdInitializer):
    """Initializer for various version of principal moment of covariance matrix.

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
                 traj_file_clst="clusters_covl.traj",
                 output_every=10, formula="I1/I3"):
        from cemc.mcmc import CovarianceMatrixObserver
        if matrix_element in cluster_elements:
            raise ValueError("CovarianceCrdInitializer works only when "
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
        self.cov_obs = CovarianceMatrixObserver(atoms=fixed_nucl_mc.atoms, cluster_elements=cluster_elements)

        # Attach the covariance matrix observer to the 
        # fixed nucleation sampler
        self.fixed_nucl_mc.attach(self.cov_obs)
        self.traj_file = traj_file
        self.traj_file_clst = traj_file_clst

    def covariance_matrix(self, atoms, system_changes):
        """Calculate the covariance matrix of the cluster.

        :return: Covariance matrix
        :rtype: Numpy 3x3 matrix
        """
        if system_changes:
            self.cov_obs(system_changes)
        elif atoms is not None:
            # Perform a new calculation from scratch
            self.cov_obs.set_atoms(atoms)

        cov = self.cov_obs.cov_matrix

        # This class should not alter the intertia tensor
        # so we undo the changes
        if system_changes:
            self.cov_obs.undo_last()
        return cov

    def principal_variance(self, atoms, system_changes):
        """Calculate the covariance of the atoms in cluster elements.

        :return: Principal variances
        :rtype: numpy 1D array of length 3
        """
        eigv = np.linalg.eigvals(self.covariance_matrix(atoms, system_changes))
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

    def normalized_principal_variance(self, atoms, system_changes):
        """Principal covariance normalized by the largest component.

        :return: Normalized principal variance
        :rtype: 1D numpy array of length 3
        """
        princ_var = self.principal_variance(atoms, system_changes)
        return princ_var / np.max(princ_var)

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
        """Get the covariance reaction coordinate.

        :param Atoms atoms: Not used. Using the atoms object of fixed_nucl_mc.

        :return: The reaction coordinate
        :rtype: float
        """
        princ = self.principal_variance(atoms, system_changes)
        princ = np.sort(princ)

        # Make sure they are sorted in the correct order
        assert princ[0] <= princ[2]

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
        self.cov_obs.init_com_and_covariance()
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
                print("Current value: {} Target value: {}"
                      "".format(new_value, value))
                sys.stdout.flush()
                now = time.time()

            if new_diff < current_diff:
                # The candidate trial moves brings the system closer to the
                # target value, so we accept this move
                current_diff = new_diff

                # We need to update the covariance observer
                self.cov_obs(system_changes)

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


class CovarianceRangeConstraint(ReactionCrdRangeConstraint):
    """Constraint to ensure that the system stays without its bounds.

    :param FixedNucleusMC fixed_nuc_mc: Monte Carlo object
    :param list range: Upper and lower bound of the reaction coordinate
    :param CovarianceCrdInitializer cov_init: Initializer
    :param bool verbose: If True print messages every 10 sec
        if the constraint is violated
    """

    def __init__(self, fixed_nuc_mc=None, range=[0.0, 1.0], cov_init=None,
                 verbose=False):
        super(CovarianceRangeConstraint, self).__init__()
        self.update_range(range)
        self.mc = fixed_nuc_mc
        self._cov_init = cov_init
        self.last_print = time.time()
        self.verbose = verbose

    def get_new_value(self, system_changes):
        """Get new value for reaction coordinate.

        :param list system_changes: List with the proposed changes

        :return: Reaction coordate after the change
        :rtype: float
        """

        # Get the new value of the observer
        new_val = self._cov_init.get(None, system_changes=system_changes)
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
            if time.time() - self.last_print > 10:
                print("Move violates constraint")
                self.last_print = time.time()
        return ok

    