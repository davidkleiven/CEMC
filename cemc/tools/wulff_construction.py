import numpy as np


class WulffConstruction(object):
    def __init__(self, cluster=None, max_dist_in_element=None):
        self.cluster = cluster
        self.max_dist_in_element = max_dist_in_element
        self.mesh = self._mesh()
        self.surf_mesh = self._extract_surface_mesh(self.mesh)
        self.linear_fit = {}
        self.spg_group = None

    def _mesh(self):
        """Create mesh of all the atoms in the cluster.

        :return: Mesh
        :rtype: scipy.spatial.Delaunay
        """
        from scipy.spatial import Delaunay
        points = self.cluster.get_positions()
        delaunay = Delaunay(points)
        simplices = self._filter_max_dist_in_element(delaunay.simplices)
        delaunay.simplices = simplices
        return delaunay

    def _filter_max_dist_in_element(self, simplices):
        """Filter out triangulations where the distance is too large."""
        if self.max_dist_in_element is None:
            return simplices

        filtered = []
        for tup in simplices:
            dists = []
            for root in tup:
                new_dist = self.cluster.get_distances(root, tup)
                dists += list(new_dist)

            if max(dists) < self.max_dist_in_element:
                filtered.append(tup)
        return filtered

    def show(self):
        """Show the triangulation in 3D."""
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        pos = self.cluster.get_positions()
        from itertools import combinations
        for tri in self.mesh.simplices:
            for comb in combinations(tri, 2):
                x1 = pos[comb[0], 0]
                x2 = pos[comb[1], 0]
                y1 = pos[comb[0], 1]
                y2 = pos[comb[1], 1]
                z1 = pos[comb[0], 2]
                z2 = pos[comb[1], 2]
                ax.plot([x1, x2], [y1, y2], zs=[z1, z2], color="black")
        plt.show()

    def _extract_surface_mesh(self, volume_mesh):
        """Extract surface cells.

        :return: List of indices belonging to the surface
        :rtype: List of tuples
        """
        from itertools import combinations
        simplices = volume_mesh.simplices
        statistics = {}
        for tri in simplices:
            for comb in combinations(tri, 3):
                key = tuple(sorted(comb))
                statistics[key] = statistics.get(key, 0) + 1

        # Trianges on surface enters only once. If they are in bulk
        # they always occures twice
        tri_surf = []
        for k, v in statistics.items():
            if v == 1:
                tri_surf.append(k)
        return tri_surf

    def _unique_surface_indices(self, surf_mesh):
        """Extract the unique positions of the atoms on the surface

        :return: List with indices of the atoms belonging to the surface
        :rtype: Atoms on the surface
        """
        flattened = []
        for tup in surf_mesh:
            flattened += list(tup)
        return list(set(flattened))

    @property
    def surface_atoms(self):
        """Return all the surface atoms."""
        indx = self._unique_surface_indices(self.surf_mesh)
        return self.cluster[indx]

    def normal_vector(self, facet):
        """Find the normal vector of the triangular facet.

        :param list facet: List with three integer describing the facet

        :return: The normal vector n = [n_x, n_y, n_z]
        :rtype: numpy 1D array of length 3
        """
        assert len(facet) == 3
        pos = self.cluster.get_positions()
        v1 = pos[facet[1], :] - pos[facet[0], :]
        v2 = pos[facet[2], :] - pos[facet[0], :]
        n = np.cross(v1, v2)
        length = np.sqrt(np.sum(n**2))
        return n / length

    @property
    def interface_energy(self):
        com = np.sum(self.cluster.get_positions(), axis=0) / len(self.cluster)

        data = []
        pos = self.cluster.get_positions()
        for facet in self.surf_mesh:
            n = self.normal_vector(facet)
            point_on_facet = pos[facet[0], :]
            vec = point_on_facet - com
            dist = vec.dot(n)

            if dist < 0.0:
                data.append((-n, -dist))
            else:
                data.append((n, dist))
        return data

    def symmetry_equivalent_directions(self, vec):
        """Return a list with all symmetry equivalent directions."""
        if self.spg_group is None:
            equiv_vec = np.zeros((1, 3))
            equiv_vec[0, :] = vec
            return equiv_vec

        rot = self.spg_group.get_rotations()
        equiv_vec = np.zeros((len(rot), 3))
        for i, mat in enumerate(rot):
            new_vec = mat.dot(vec)

            # TODO: Why is this needed? The length should be conserved
            new_vec /= np.sqrt(new_vec.dot(new_vec))
            equiv_vec[i, :] = new_vec
        return equiv_vec

    def _unique_columns(self, A):
        """Get a list of unique columns."""
        equal_columns = []
        for i in range(0, A.shape[1]):
            for j in range(i+1, A.shape[1]):
                if np.allclose(A[:, i], A[:, j]):
                    equal_columns.append((i, j))

        # In addition constant columns are equal to the first column
        equal_due_to_constant = []
        for i in range(1, A.shape[1]):
            if np.allclose(A[:, i], A[0, i]):
                equal_due_to_constant.append((0, i))

        equal_columns = equal_due_to_constant + equal_columns
        unique_columns = list(range(A.shape[1]))
        for eq in equal_columns:
            if eq[0] in unique_columns and eq[1] in unique_columns:
                unique_columns.remove(eq[1])
        return unique_columns

    def _get_x_value(self, vec, comb):
        """Return the x value in the polynomial expansion."""
        eq_vec = self.symmetry_equivalent_directions(vec)
        x_avg = 0.0
        for vec_indx in range(eq_vec.shape[0]):
            v = eq_vec[vec_indx, :]
            x = 1.0
            for indx in comb:
                x *= v[indx]
            x_avg += x
        return x_avg / eq_vec.shape[0]

    def interface_energy_poly_expansion(self, order=2, show=False, spg=1):
        """Fit a multidimensional polynomial of a certain order."""
        from itertools import combinations_with_replacement
        interf = self.interface_energy

        if spg > 1:
            from ase.spacegroup import Spacegroup
            self.spg_group = Spacegroup(spg)
        num_terms = int((3**(order+1) - 1)/2)

        A = np.zeros((len(interf), num_terms))
        A[:, 0] = 1.0
        col = 1
        mult_order = [()]
        for p in range(1, order+1):
            for comb in combinations_with_replacement(range(3), p):
                print(comb)
                vec = np.zeros(len(interf))
                row = 0
                for n, value in interf:
                    x = self._get_x_value(n, comb)
                    vec[row] = x
                    row += 1
                A[:, col] = vec
                mult_order.append(comb)
                col += 1
        rhs = np.zeros(len(interf))
        row = 0
        for n, value in interf:
            rhs[row] = value
            row += 1

        unique_cols = self._unique_columns(A)
        A = A[:, unique_cols]
        mult_order = [mult_order[indx] for indx in unique_cols]
        coeff, residual, rank, s = np.linalg.lstsq(A, rhs)
        print(coeff)
        self.linear_fit["coeff"] = coeff
        self.linear_fit["order"] = mult_order
        if show:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            pred = A.dot(coeff)
            ax.plot(pred, rhs, 'o', mfc="none")
            plt.show()

    def eval(self, theta, phi):
        """Evaluate interface energy in a particular direction using fit.

        :param float theta: Polar angle in radians
        :param float phi: Azimuthal angle in radian

        :return: The estimated interface value
        :rtype: float
        """
        required_fields = ["coeff", "order"]
        for field in required_fields:
            if field not in self.linear_fit.keys():
                raise ValueError("It looks like "
                                 "interface_energy_poly_expansion "
                                 "has not been called. Call that function "
                                 "first.")

        res = self.linear_fit["coeff"][0]
        loop = zip(self.linear_fit["order"][1:],
                   self.linear_fit["coeff"][1:].tolist())
        n = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
             np.cos(theta)]
        for order, coeff in loop:
            x = self._get_x_value(n, order)
            res += coeff*x
        return res

    def wulff_plot(self, show=False, n_angles=120):
        """Create a Wulff plot."""
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig_xy = plt.figure()
        ax_xy = fig_xy.add_subplot(1, 1, 1)
        pos = self.cluster.get_positions()
        com = np.mean(pos, axis=0)
        pos -= com

        # Project atomic positions into the xy plane
        proj_xy = pos[:, :2]
        ax_xy.plot(proj_xy[:, 0], proj_xy[:, 1], 'x')
        n_angles = 100
        theta = np.zeros(n_angles) + np.pi/2.0
        theta = theta.tolist()
        phi = np.linspace(0.0, 2.0*np.pi, n_angles).tolist()
        gamma = np.array([self.eval(t, p) for t, p in zip(theta, phi)])
        x = gamma * np.cos(phi)
        y = gamma * np.sin(phi)
        ax_xy.plot(x, y)

        # Plot the full surface in 3D
        try:
            from itertools import product
            from mayavi import mlab
            theta = np.linspace(0.0, np.pi, n_angles)
            phi = np.linspace(0.0, 2.0*np.pi, n_angles)
            theta = theta.tolist()
            T, P = np.meshgrid(theta, phi)
            Gamma = np.zeros(T.shape)
            for indx in product(range(n_angles), range(n_angles)):
                Gamma[indx] = self.eval(T[indx], P[indx])

            X = Gamma*np.cos(P)*np.sin(T)
            Y = Gamma*np.sin(P)*np.sin(T)
            Z = Gamma*np.cos(T)
            mlab.mesh(X, Y, Z, scalars=Gamma)
            if show:
                mlab.show()
        except ImportError as exc:
            print("{}: {}".format(type(exc).__name__, str(exc)))
            print("To visualize in 3D mayavi is required!")

        if show:
            plt.show()
        return fig_xy
