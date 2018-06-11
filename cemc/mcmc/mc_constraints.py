
class MCConstraint(object):
    """
    Class for that prevents the MC sampler to run certain moves
    """
    def __init__(self):
        self.name = "GenericConstraint"

    def __call__(self, system_changes):
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
        super(PairConstraint,self).__init__()
        self.name = "PairConstraint"

        if calc is None:
            raise ValueError("No calculator object given!")
        elif cluster_name is None:
            raise ValueError("No cluster name given!")
        elif elements is None:
            raise ValueError("No element list given!")

        size = int(cluster_name[1])
        if size != 2:
            raise ValueError("Only pair clusters given. Given cluster has {} elements".format(size))

        if len(elements) != 2:
            raise ValueError("The elements list has to consist of exactly two elements")
        self.cluster_name = elements
        self.elements = elements

        elem1 = str(self.elements[0])
        elem2 = str(self.elements[1])
        cname = str(self.cluster_name)
        self.cpp_constraint = PairConstCpp(calc.updater, cname, \
            elem1, elem2)

    def __call__(self, system_changes):
        """
        Checks if there are any pairs of the two atoms.
        """

        # Force the calculator to update the symbols
        return not self.cpp_constraint.elems_in_pair(system_changes)
