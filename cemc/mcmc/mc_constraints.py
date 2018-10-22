import numpy as np

class MCConstraint(object):
    """
    Class for that prevents the MC sampler to run certain moves
    """

    def __init__(self):
        self.name = "GenericConstraint"

    def __call__(self, system_changes):
        """Return true if the trial move is valid.

        :param list system_changes: List of tuples with information about the
                               changes introduced. See doc string of
                               :py:class:`cemc.mcmc.mc_observers.MCObserver`

        :return: True/False, if True the move is valid
        :rtype: bool
        """
        return True


class PairConstraint(MCConstraint):
    """
    Prevent two atoms to be in a pair cluster

    :param Atoms atoms: Atoms object in the MC simulaton
    :param str cluster_name: Name of cluster that the pair cannot enter in
    :param list elements: Elements not supposed to enter in a cluster
    """

    def __init__(self, calc=None, cluster_name=None, elements=None):
        from cemc_cpp_code import PyPairConstraint
        super(PairConstraint, self).__init__()
        self.name = "PairConstraint"
        self.calc = calc
        self.cluster_name = cluster_name
        self.elements = elements

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
        self.cpp_constraint = PyPairConstraint(calc.updater, cname, elem1, elem2)

    def __reduce__(self):
        return (self.__class__, (self.calc, self.cluster_name, self.elements))

    def __call__(self, system_changes):
        """
        Check if there are any pairs of the two atoms.

        :param list system_changes: Proposed changes

        :return: True/False if constraint is violated
        :rtype: bool
        """

        # Force the calculator to update the symbols
        return not self.cpp_constraint.elems_in_pair(system_changes)


class FixedElement(MCConstraint):
    """
    Prevents an element to be moved

    :param str element: Element to fix
    """

    def __init__(self, element=None):
        super(FixedElement, self).__init__()
        self.name = "FixedElement"
        self.element = element

    def __call__(self, system_changes):
        """
        Check if the *element* is involved in any of the changes

        :param list system_changes: Proposed changes

        :return: True/False, if True, the move is valid
        :rtype: bool
        """
        for change in system_changes:
            if change[0] == self.element or change[1] == self.element:
                return False
        return True


class FixEdgeLayers(MCConstraint):
    """Fix atoms near border of the cell."""
    def __init__(self, thickness=4.0, atoms=None):
        self.cell = atoms.get_cell()
        self.thickness = thickness
        self.planes = self._init_planes()
        self.atoms = atoms
        self.pos = self.atoms.get_positions()

    def _init_planes(self):
        """Initialize the planes."""
        from cemc.tools.geometry import Plane
        from itertools import combinations
        cross_comb = [(0, 1), (0, 2), (1, 2)]
        trans_vec = [2, 1, 0]
        planes = []
        for t, comb in zip(trans_vec, cross_comb):
            normal = np.cross(self.cell[comb[0], :], self.cell[comb[1], :])
            unit_vec = self.cell[t, :]/np.sqrt(np.sum(self.cell[t, :]**2))
            pts_in_plane = np.array(-0.1*self.thickness*unit_vec)
            p1 = Plane(normal, pts_in_plane)

            pts_in_plane = unit_vec*self.thickness
            p2 = Plane(normal, pts_in_plane)
            planes.append((p1, p2))

            pts_in_plane = self.cell[t, :] + 0.1*self.thickness*unit_vec
            p3 = Plane(normal, pts_in_plane)
            pts_in_plane = self.cell[t, :] - self.thickness*unit_vec
            p4 = Plane(normal, pts_in_plane)
            planes.append((p3, p4))
        return planes

    def _is_between_planes(self, x, p1, p2):
        """Check if a point is between two planes."""
        d1 = p1.signed_distance(x)
        d2 = p2.signed_distance(x)
        return np.sign(d1*d2) < 0

    def _is_between_any(self, x):
        """Check if a point is between any pairs of planes."""
        for p in self.planes:
            if self._is_between_planes(x, p[0], p[1]):
                return True
        return False

    def __call__(self, system_changes):
        """Check if move is valid
        Return True if the move is valid
        """
        for change in system_changes:
            x = self.pos[change[0], :]
            if self._is_between_any(x):
                return False
        return True

    def get_fixed_atoms(self):
        """Return an atoms object containing only the fixed sites."""
        indices = []
        for i in range(self.pos.shape[0]):
            if self._is_between_any(self.pos[i, :]):
                indices.append(i)
        return self.atoms[indices]
        
