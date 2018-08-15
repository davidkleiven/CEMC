from cemc.mcmc import SGCMonteCarlo
import numpy as np
from random import choice, shuffle


class PseudoBinarySGC(SGCMonteCarlo):
    """
    Class that can assign a chemical potential to a group of atoms

    :param atoma: Atoms object
    :param T: Temperature
    :param kwargs: See :py:class:`cemc.mcmc.SGCMonteCarlo`
        Special key word arguments for this class
        * group: The pseudo binary groups [{"Al": 2}, {"Mg": 1, "Si": 1}],
                 will allow for inserting MgSi for 2 Al atoms
        * insert_prob: Probability to try an insertion of the pseudo binary
                       groups
        * chem_pot: Chemical potential associated with the pseudo-binary groups
    """

    def __init__(self, atoms, T, **kwargs):
        self._groups = kwargs.pop("group")
        self._check_group()
        self._chem_pot = kwargs.pop("chem_pot")
        self._chem_pot_dict = self._get_eci_chem_pot()
        self._ins_prob = 0.1
        if "insert_prob" in kwargs.keys():
            self._ins_prob = kwargs.pop("insert_prob")
        SGCMonteCarlo.__init__(self, atoms, T, **kwargs)

        # This class needs to track the positions of the elemsnts,
        # but since the number of elements of each symbol is not constant,
        # we need apply a different datastructure
        self.atoms_indx = self._init_symbol_tracker()

    def _init_symbol_tracker(self):
        """Initialize the symbol tracker."""
        # Initialize with an empty set
        atoms_indx = {symb: set([]) for symb in self.symbols}

        # Populate the sets
        for atom in self.atoms:
            symb = atom.symbol
            atoms_indx[symb].add(atom.index)
        return atoms_indx

    def _check_group(self):
        """Check that the group argument given by the user is fine."""
        if len(self._groups) != 2:
            raise ValueError("There have to be two groups!")

        # Check the number of atoms in each group is the same
        n_group1 = 0
        for key, value in self._groups[0].items():
            n_group1 += value

        n_group2 = 0
        for key, value in self._groups[1].items():
            n_group2 += value

        if n_group1 != n_group2:
            f1 = self._group2formula(self._groups[0])
            f2 = self._group2formula(self._groups[1])
            msg = "The two groups have to have the same number of atoms.\n"
            msg += "Group 1: {} Group 2: {}".format(f1, f2)
            raise ValueError(msg)

    def _group2formula(self, elem_dict):
        """Convert a dictionary description into a string description."""
        formula = ""
        for key, value in elem_dict:
            formula += "{}{}".format(key, value)
        return formula

    def _get_eci_chem_pot(self):
        """Compute the chemical potentials we need to add to the ECIs."""
        bf = self.atoms.get_calculator().BC.basis_functions
        bf_change_vec = np.zeros((1, len(bf)))
        for i, func in enumerate(bf):
            for key, num in self._groups[0].items():
                bf_change_vec[0, i] += func[key] * num

            for key, num in self._groups[1].items():
                bf_change_vec[0, i] -= func[key] * num
        pinv = np.linalg.pinv(bf_change_vec)
        mu_vec = pinv.dot(np.array([self._chem_pot]))

        chem_pot_dict = {}
        for i in range(len(mu_vec)):
            chem_pot_dict["c1_{}".format(i)] = mu_vec
        return chem_pot_dict

    def _insert_trial_move(self):
        """Trial move consisting of inserting a new unit of pseudo binary."""
        syst_changes = []
        grp_indx = [0, 1]
        shuffle(grp_indx)
        grp1 = self._groups[grp_indx[0]]
        grp2 = self._groups[grp_indx[1]]

        # Find indices of elements in the first group
        for key, num in grp1.items():
            indices = self._get_random_index(key, num=num)
            for indx in indices:
                syst_changes.append((indx, key, None))

        # Update with symbols from the other group
        sys_ch = 0
        for key, num in grp2.items():
            for _ in range(num):
                syst_changes[sys_ch][2] = key
                sys_ch += 1
        return sys_ch

    def _get_random_index(self, symbol, num=1):
        """Get a random index of an atom with given symbol."""
        # NOTE: pop removes an arbitrary element
        indices = []
        for _ in range(num):
            indices.append(self.atoms_indx[symbol].pop())

        # Insert it back again to not alter the tracker
        self.atoms_indx[symbol].update(indices)
        return indices

    def _swap_trial_move(self):
        """Swap two atoms."""
        symb1 = choice(self.symbols)
        symb2 = symb1
        while symb2 == symb1:
            symb2 = choice(self.symbol)

        syst_changes = []
        indx = self._get_random_index(symb1)[0]
        syst_changes.append((indx, symb1, symb2))

        indx = self._get_random_index(symb2)[0]
        syst_changes.append((indx, symb2, symb1))
        return syst_changes

    def _update_tracker(self, changes):
        """Update the atom tracker, only called if moves are accepted."""
        for change in changes:
            indx = change[0]
            old_symb = change[1]
            new_symb = change[2]
            self.atoms_indx[old_symb].remove(indx)
            self.atoms_indx[new_symb].add(indx)

    def _get_trial_move(self):
        """Calculate a trial move."""
        max_attempts = 1000
        if np.random.rand() < self._ins_prob:
            try:
                change = self._insert_trial_move()
                count = 0
                while (not self._no_constraint_violations(change)
                       and count < max_attempts):
                    change = self._insert_trial_move()
                    count += 1

                if count < max_attempts:
                    return change
            except KeyError:
                # Could not generate an insertion move
                pass

        change = self._swap_trial_move()
        count = 0
        while (not self._no_constraint_violations(change)
               and count < max_attempts):
            change = self._swap_trial_move()
            count += 1

        if count == max_attempts:
            msg = "Did not manage to find a valid move "
            msg += "in {} attemps!".format(max_attempts)
            raise RuntimeError(msg)
        return change

    def runMC(self, **kwargs):
        """Run Monte Carlo simulation.
          See :py:func:`cemc.mcmc.SGCMonteCarlo.runMC`
        """
        if "chem_potential" in kwargs.keys():
            kwargs.pop("chem_potential")
        kwargs["chem_potential"] = self._chem_pot_dict
        SGCMonteCarlo.runMC(self, **kwargs)
