import numpy as np
import time

class WulffConstruction(object):
    def __init__(self, cluster=None, max_dist_in_element=None):
        self.cluster = cluster
        self.max_dist_in_element = max_dist_in_element
        self.mesh = self._mesh()
        self.surf_mesh = self._extract_surface_mesh(self.mesh)
        self.linear_fit = {}
        self.spg_group = None
        self.angles = []
        self._interface_energy = []

    def filter_neighbours(self, num_neighbours=1, elements=None, cutoff=6.0):
        """Remove atoms that has to few neighbours within the specified cutoff.

        :param int num_neighbours: Number of required neighbours
        :param elements: Which elements are considered part of the cluster
        :type elements: list of strings
        :param float cutoff: Cut-off radius in angstrom
        """
        from scipy.spatial import cKDTree as KDTree
        if elements is None:
            raise TypeError("No elements given!")

        print("Filtering out atoms with less than {} elements "
              "of type {} within {} angstrom"
              "".format(num_neighbours, elements, cutoff))

        tree = KDTree(self.cluster.get_positions())
        indices_in_cluster = []
        for atom in self.cluster:
            neighbours = tree.query_ball_point(atom.position, cutoff)
            num_of_correct_symbol = 0
            for neigh in neighbours:
                if self.cluster[neigh].symbol in elements:
                    num_of_correct_symbol += 1
            if num_of_correct_symbol >= num_neighbours:
                indices_in_cluster.append(atom.index)
        
        self.cluster = self.cluster[indices_in_cluster]
        # Have to remesh everything
        self.mesh = self._mesh()
        self.surf_mesh = self._extract_surface_mesh(self.mesh)

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


    def save_surface_mesh(self, fname):
        """Save the surface mesh to GMSH format

        :param str fname: Filename to where mesh will be stored
        """
        triangulation = self.surf_mesh
        points = self.cluster.get_positions()
        with open(fname, 'w') as out:
            # Write mandatory header
            out.write("$MeshFormat\n")
            out.write("2.2 0 8\n")
            out.write("$EndMeshFormat\n\n")


            # Write points
            out.write("$Nodes\n")
            out.write("{}\n".format(points.shape[0]))
            for i in range(points.shape[0]):
                vec = points[i, :]
                out.write("{} {} {} {}\n".format(i+1, vec[0], vec[1], vec[2]))
            out.write("$EndNodes\n")

            # Write triangles
            out.write("$Elements\n")
            out.write("{}\n".format(len(triangulation)))
            for i, tri in enumerate(triangulation):
                out.write("{} 2 0 {} {} {}\n".format(i+1, tri[0]+1, tri[1]+1, tri[2]+1))
            out.write("$EndElements\n")

            if len(self.angles) == len(triangulation):
                # Angles between the normal vector of the facet and the 
                # centroid has been computed. We store this as element data
                out.write("$ElementData\n")
                out.write("1\n")
                out.write("NormalVectorAngle\n")
                out.write("0\n")
                out.write("4\n0\n1\n{}\n0\n".format(len(self.angles)))
                for i, ang in enumerate(self.angles):
                    angle = ang
                    if angle > 90.0:
                        angle = 180.0 - angle
                    out.write("{} {}\n".format(i+1, angle))
                out.write("$EndElementData\n")

            if len(self._interface_energy) == len(triangulation):
                # Interface energy has been computed
                # We store this as element data
                out.write("$ElementData\n")
                out.write("1\n")
                out.write("Gamma\n")
                out.write("0\n")
                out.write("4\n0\n1\n{}\n0\n".format(len(self._interface_energy)))
                for i, interf in enumerate(self._interface_energy):
                    out.write("{} {}\n".format(i+1, interf[1]))
                out.write("$EndElementData\n")
        print("Surface mesh saved to {}".format(fname))

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
        com = np.mean(self.cluster.get_positions(), axis=0)

        data = []
        pos = self.cluster.get_positions()
        self.angles = []
        for facet in self.surf_mesh:
            n = self.normal_vector(facet)

            # Calculate centroid of the facet
            point_on_facet = (pos[facet[0], :] + pos[facet[1], :] + pos[facet[2], :])/3.0
            vec = point_on_facet - com
            dist = vec.dot(n)
            angle = np.arccos(dist/np.sqrt(vec.dot(vec)))*180.0/np.pi
            if angle > 90.0:
                angle = 180.0 - angle
            self.angles.append(angle)

            if dist < 0.0:
                data.append((-n, -dist))
            else:
                data.append((n, dist))
        self._interface_energy = data
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

    def interface_energy_poly_expansion(self, order=2, show=False, spg=1,
                                        penalty=0.0, max_angle=90):
        """Fit a multidimensional polynomial of a certain order."""
        from itertools import combinations_with_replacement
        interf = self.interface_energy

        # Filter out the surfaces that has extremely high angles
        inter_filtered = []
        for data, angle in zip(interf, self.angles):
            if angle < max_angle:
                inter_filtered.append(data)
        interf = inter_filtered

        if spg > 1:
            from ase.spacegroup import Spacegroup
            self.spg_group = Spacegroup(spg)
        num_terms = int((3**(order+1) - 1)/2)

        A = np.zeros((len(interf), num_terms))
        A[:, 0] = 1.0
        col = 1
        mult_order = [()]
        now = time.time()
        output_every = 10
        for p in range(1, order+1):
            for comb in combinations_with_replacement(range(3), p):
                if time.time() - now > output_every:
                    print("Calculating order {} permutation {}"
                          "".format(p, comb))
                    now = time.time()
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

        print("Filtering duplicates...")
        unique_cols = self._unique_columns(A)
        A = A[:, unique_cols]
        mult_order = [mult_order[indx] for indx in unique_cols]
        print("Solving linear system...")
        if A.shape[1] == 1:
            # TODO: Trivial solution, calculate this directly
            coeff, residual, rank, s = np.linalg.lstsq(A, rhs)
        else:
            N = A.shape[1]
            matrix = np.linalg.inv(A.T.dot(A) + penalty*np.identity(N))
            coeff = matrix.dot(A.T.dot(rhs))
        self.linear_fit["coeff"] = coeff
        self.linear_fit["order"] = mult_order
        pred = A.dot(coeff)
        rmse = np.sqrt(np.mean((pred-rhs)**2))
        print("RMSE of surface parametrization: {}".format(rmse))
        if show:
            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
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

    def fit_harmonics(self, show=False, order=3, penalty=0.0):
        """Fit a spherical harmonics expansion to the surface."""
        pts = self.surface_atoms.get_positions()
        com = np.mean(pts, axis=0)
        pts -= com
        r = np.sqrt(np.sum(pts**2, axis=1))
        theta = np.arccos(pts[:, 2]/r)
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        data = np.zeros((len(phi), 3))
        data[:, 0] = phi
        data[:, 1] = theta
        data[:, 2] = r


        from cemc.tools import HarmonicsFit
        fit = HarmonicsFit(order=order)
        fit.fit(data, penalty=penalty)
        if show:
            fit.show()
        return fit

    def wulff_plot(self, show=False, n_angles=120):
        """Create a Wulff plot."""
        from matplotlib import pyplot as plt
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
            print("Evaluating gamma at all angles...")
            for indx in product(range(n_angles), range(n_angles)):
                Gamma[indx] = self.eval(T[indx], P[indx])

            X = Gamma*np.cos(P)*np.sin(T)
            Y = Gamma*np.sin(P)*np.sin(T)
            Z = Gamma*np.cos(T)
            fig = mlab.figure(bgcolor=(1, 1, 1))
            mlab.mesh(X, Y, Z, scalars=Gamma/np.min(Gamma))
            if show:
                mlab.show()
        except ImportError as exc:
            print("{}: {}".format(type(exc).__name__, str(exc)))
            print("To visualize in 3D mayavi is required!")

        if show:
            plt.show()
        return fig_xy
