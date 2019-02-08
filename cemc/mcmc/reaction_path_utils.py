from cemc.mcmc import MCConstraint


class CannotInitSysteError(Exception):
    pass


class ReactionCrdRangeConstraint(MCConstraint):
    """
    Generic class for range constraint. Parent of all other
    range constraints.
    """
    def __init__(self):
        MCConstraint.__init__(self)
        self.name = "GenericReactionCrdRangeConstraint"
        self.range = [0.0, 1.0]

    def __call__(self, system_changes):
        """Check if atoms object is inside the constraint."""
        raise NotImplementedError("Has to be implemented in derived classes!")

    def update_range(self, new_range):
        """Update the range."""
        self.range = new_range


class ReactionCrdInitializer(object):
    """
    Class for bringing system into a certain state.
    """

    def __init__(self):
        pass

    def set(self, atoms, value):
        """Bring the atoms into a state with given value of reaction crd.

        :param atoms: An atoms object
        :param value: Value of the reaction coordinate
        """
        raise NotImplementedError("Has to be implemented in derived classes!")

    def get(self, atoms, system_changes=[]):
        """Get the current value of the reaction coordinate.

        :param atoms: An atoms object
        """
        raise NotImplementedError("Has to be implemented in derived classes!")


class PseudoBinaryConcInitializer(ReactionCrdInitializer):
    """
    Sets an arbitrary concentration of the pseudo binary group 2

    :param pseudo_mc: Instance of
                      :py:class:`cemc.mcmc.pseudo_binary_mc.PseudoBinarySGC`
    """

    def __init__(self, pseudo_mc):
        ReactionCrdInitializer.__init__(self)
        self.name = "PseudoBinaryConcInitializer"
        self.mc = pseudo_mc
        self.target_symb = list(self.mc.groups[1].keys())[0]

    def set(self, atoms, number):
        """Introduce a given number of pseudo binary elements.

        :param atoms: An atoms object
        :param number: Number of units of pseudo binary group 2
        """
        group = self.mc.groups
        num_units = float(len(self.mc.atoms_indx[self.target_symb])) / \
            group[1][self.target_symb]

        grp1 = sorted(self.mc._symbs_in_group_in_random_order(group[0]))
        grp2 = sorted(self.mc._symbs_in_group_in_random_order(group[1]))

        # Calculate number of formula units to insert/remove
        num_insert = number - num_units

        num_inserted = 0
        max_attempts = 10000
        count = 0
        while num_inserted < abs(num_insert) and count < max_attempts:
            try:
                change = self.mc.insert_trial_move()
            except KeyError:
                count += 1
                continue
            count += 1

            new_symb = sorted([item[2] for item in change])
            if num_insert > 0:
                # Should increase number of group 2
                # make sure the new symbols belongs to group 2
                valid_move = new_symb == grp2
            else:
                valid_move = new_symb == grp1

            if valid_move and self.mc._no_constraint_violations(change):
                calc = self.mc.atoms.get_calculator()
                calc.calculate(self.mc.atoms, ["energy"], change)
                self.mc._update_tracker(change)
                calc.clear_history()
                num_inserted += 1

        if count == max_attempts:
            msg = "Could not initialize the system in "
            msg += "{} attempts".format(max_attempts)
            raise CannotInitSysteError(msg)

    def get(self, atoms, system_changes=None):
        """Get the number of formula units of group 2.

        :param atoms: An atoms object
        """
        if system_changes is None:
            num_per_unit = self.mc.groups[1][self.target_symb]
            return float(len(self.mc.atoms_indx[self.target_symb])) / num_per_unit

        return self.get_value_from_syst_change(system_changes)

    def get_value_from_syst_change(self, syst_changes): 
        n_un = self.number_of_units
        for change in syst_changes:
            if change[2] == self.target_symb:
                n_un += 1.0 / self.num_per_unit
            elif change[1] == self.target_symb:
                n_un -= 1.0 / self.num_per_unit
        return n_un

    @property
    def number_of_units(self):
        return float(len(self.mc.atoms_indx[self.target_symb])) / \
            self.num_per_unit

    @property
    def num_per_unit(self):
        return self.mc.groups[1][self.target_symb]


class PseudoBinaryConcRange(ReactionCrdRangeConstraint):
    """
    Range constraint for the number of units of pseudo binary group 2.

    :param mc_obj: Instance of
                   :py:class:`cemc.mcmc.pseudo_binary_mc.PseudoBinarySGC`
    """

    def __init__(self, mc_obj):
        ReactionCrdRangeConstraint.__init__(self)
        self.name = "PseudoBinaryConcRange"
        self.mc = mc_obj
        self.target_symb = list(self.mc.groups[1].keys())[0]

    @property
    def number_of_units(self):
        return float(len(self.mc.atoms_indx[self.target_symb])) / \
            self.num_per_unit

    @property
    def num_per_unit(self):
        return self.mc.groups[1][self.target_symb]

    def __call__(self, syst_changes):
        """
        Check if the system is inside the concentration range.

        :param system_changes: List of tuples describing the system changes.
                               See
                               :py:class:`cemc.mcmc.mc_observers.MCObservers`
        """
        n_un = self.number_of_units
        for change in syst_changes:
            if change[2] == self.target_symb:
                n_un += 1.0 / self.num_per_unit
            elif change[1] == self.target_symb:
                n_un -= 1.0 / self.num_per_unit
        return n_un >= self.range[0] and n_un < self.range[1]
