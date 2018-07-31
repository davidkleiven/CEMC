
class MCConstraint(object):
    """
    Class for that prevents the MC sampler to run certain moves
    """

    def __init__(self):
        self.name = "GenericConstraint"

    def __call__(self, system_changes):
        """Return true if the trial move is valid.

        :param system_changes: List of tuples with information about the
                               changes introduced. See doc string of
                               :py:class:`cemc.mcmc.mc_observers.MCObserver`
        """
        return True


class PairConstraint(MCConstraint):
    """
    Prevent two atoms to be in a pair cluster

    :param atoms: Atoms object in the MC simulaton
    :param cluster_name: Name of cluster that the pair cannot enter in
    :param elements: List of elements not supposed to enter in a cluster
    """

    def __init__(self, calc=None, cluster_name=None, elements=None):
        from cemc.ce_updater.ce_updater import PairConstraint as PairConstCpp
        super(PairConstraint, self).__init__()
        self.name = "PairConstraint"

        if calc is None:
            raise ValueError("No calculator object given!")
        elif cluster_name is None:
            raise ValueError("No cluster name given!")
        elif elements is None:
            raise ValueError("No element list given!")

        size = int(cluster_name[1])
        if size != 2:
            msg = "Only pair clusters given."
            msg += "Given cluster has {} elements".format(size)
            raise ValueError(msg)

        if len(elements) != 2:
            msg = "The elements list has to consist of "
            msg += "exactly two elements"
            raise ValueError(msg)
        self.cluster_name = elements
        self.elements = elements

        elem1 = str(self.elements[0])
        elem2 = str(self.elements[1])
        cname = str(self.cluster_name)
        self.cpp_constraint = PairConstCpp(calc.updater, cname, elem1, elem2)

    def __call__(self, system_changes):
        """
        Check if there are any pairs of the two atoms.
        """

        # Force the calculator to update the symbols
        return not self.cpp_constraint.elems_in_pair(system_changes)


class FixedElement(MCConstraint):
    """
    Prevents an element to be moved

    :param element: Element to fix
    """

    def __init__(self, element=None):
        super(FixedElement, self).__init__()
        self.name = "FixedElement"
        self.element = element

    def __call__(self, system_changes):
        """
        Check if the *element* is involved in any of the changes
        """
        for change in system_changes:
            if change[0] == self.element or change[1] == self.element:
                return False
        return True
