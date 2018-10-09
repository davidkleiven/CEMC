import numpy as np
import time

class WulffConstruction(object):
    """Class for performing the inverse Wulff construction

    :param Atoms cluster: Atoms in a cluster
    :param float max_dist_in_element: Maximum distance between
        atoms in an tetrahedron in the triangulation.
    """
    def __init__(self, cluster=None, max_dist_in_element=None):
        self.cluster = cluster
        self.max_dist_in_element = max_dist_in_element
        self.mesh = self._mesh()
        self.surf_mesh = self._extract_surface_mesh(self.mesh)
        self.linear_fit = {}
        self.spg_group = None
        self.spg_num = 1
        self.angles = []
        self._interface_energy = []
        self._symmetry_file = "symmetry"

    @property
    def symmetry_fname(self):
        return self._symmetry_file + "{}.json".format(self.spg_num)

    @symmetry_fname.setter
    def symmetry_fname(self, fname_without_extension):
        self._symmetry_file = fname_without_extension

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
        surf_indx = self.surface_indices
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

            if self._interface_energy:
                # Interface energy has been computed
                # We store the values as node data
                out.write("$NodeData\n")
                out.write("1\n")
                out.write("\"Gamma\"\n")
                out.write("1\n0.0\n")
                out.write("4\n0\n1\n{}\n0\n".format(len(self._interface_energy)))
                for indx, interf in zip(surf_indx, self._interface_energy):
                    out.write("{} {}\n".format(indx+1, interf[1]))
                out.write("$EndNodeData\n")
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
        indx = self.surface_indices
        return self.cluster[indx]

    @property
    def surface_indices(self):
        """Return the indices of the atoms on the surface."""
        return self._unique_surface_indices(self.surf_mesh)

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

    def interface_energy(self, average_cutoff=10.0):
        from cemc.tools.normal_vector import NormalVectorEstimator
        com = np.mean(self.cluster.get_positions(), axis=0)

        data = []
        pos = self.cluster.get_positions()
        self.angles = []
        normal_estimate = NormalVectorEstimator(self.surf_mesh, pos)
        pos = self.surface_atoms.get_positions()
        for i in range(pos.shape[0]):
            n = normal_estimate.get_normal(pos[i, :], cutoff=average_cutoff)
            vec = pos[i, :] - com
            dist = vec.dot(n)
            data.append((n, dist))
        self._interface_energy = data
        normal_estimate.show_statistics()
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

    def is_valid(self, comb):
        """Screen out combinations that are impossible due to symmetry."""
        # Random direction
        n = np.random.rand(3)
        x = self._get_x_value(n, comb)
        return not np.allclose(x, 0.0)

    def _unique_columns(self, A):
        """Get a list of unique columns."""
        rand_vec = np.random.rand(A.shape[0])
        dot_prod = rand_vec.dot(A)
        unique, index = np.unique(dot_prod, return_index=True)
        return index

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

    def _save_valid_order(self, order):
        """Save a list of valid orders for a given spacegroup."""
        import json
        data = {}
        for item in order:
            key = len(item)
            if key not in data.keys():
                data[key] = [item]
            else:
                data[key].append(item)
        known_terms = list(data.keys())
        max_key = np.max(known_terms)
        for indx in range(max_key):
            if indx not in known_terms:
                # There are no terms with this order
                # We store it as an empty list
                data[indx] = []
        with open(self.symmetry_fname, 'w') as outfile:
            json.dump(data, outfile)
        print("Valid terms written to {}".format(self.symmetry_fname))

    def interface_energy_poly_expansion(self, order=2, show=False, spg=1,
                                        penalty=0.0, average_cutoff=10.0):
        """Fit a multidimensional polynomial of a certain order."""
        from itertools import combinations_with_replacement
        interf = self.interface_energy(average_cutoff=average_cutoff)
        self.spg = spg
        if spg > 1:
            from ase.spacegroup import Spacegroup
            self.spg_group = Spacegroup(spg)
        num_terms = int((3**(order+1) - 1)/2)

        print("Number of terms in polynomial expansion: {}".format(num_terms))
        # A = np.zeros((len(interf), num_terms))
        A = []
        num_data_points = len(interf)
        A.append(np.ones(num_data_points))
        col = 1
        mult_order = [()]
        now = time.time()
        output_every = 10
        self.spg_num = spg

        # Try to load already computed orders
        try:
            import json
            with open(self.symmetry_fname, 'r') as infile:
                precomputed_order = json.load(infile)
            print("Valid terms are read from {}. "
                  "Delete the file if you want a "
                  "new calculation from scratch."
                  "".format(self.symmetry_fname))
        except IOError:
            precomputed_order = {}

        pre_computed_sizes = [int(key) for key in precomputed_order.keys()]
        for p in range(1, order+1):
            if p in pre_computed_sizes:
                # Use only the ones that already have been
                # calculated
                combs = precomputed_order[str(p)]
            else:
                # No precalculations was made
                # Check all combinations
                combs = combinations_with_replacement(range(3), p)
            for comb in combs:
                if not self.is_valid(comb):
                    continue
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
                A.append(vec)
                mult_order.append(comb)
                col += 1
        A = np.array(A).T
        print(A.shape)
    
        rhs = np.zeros(len(interf))
        row = 0
        for n, value in interf:
            rhs[row] = value
            row += 1

        print("Filtering duplicates...")
        # unique_cols = self._unique_columns(A)
        # A = A[:, unique_cols]
        num_rows = A.shape[0]
        # A, unique_cols = np.unique(A, axis=1, return_index=True)
        unique_cols = self._unique_columns(A)
        A = A[:, unique_cols]
        assert A.shape[0] == num_rows
        mult_order = [mult_order[indx] for indx in unique_cols]

        # Filter constant columns
        constant_columns = []
        for i in range(1, A.shape[1]):
            if np.allclose(A[:, i], A[0, i]):
                constant_columns.append(i)
        A = np.delete(A, constant_columns, axis=1)
        assert A.shape[0] == num_rows

        mult_order = [mult_order[indx] for indx in range(len(mult_order)) if indx not in constant_columns]
        self._save_valid_order(mult_order)
        print("Number of terms after applying spacegroup symmetries: {}".format(A.shape[1]))
        print("Solving linear system...")
        if A.shape[1] == 1:
            # TODO: Trivial solution, calculate this directly
            coeff, residual, rank, s = np.linalg.lstsq(A, rhs)
        else:
            N = A.shape[1]
            matrix = np.linalg.inv(A.T.dot(A) + penalty*np.identity(N))
            coeff = matrix.dot(A.T.dot(rhs))

        # Perform one consistency check
        mean_val = np.mean(rhs)
        if coeff[0] > 2.0*mean_val or coeff[0] < 0.5*mean_val:
            print("Warning! This fit looks suspicious. Constant term {}"
                  "Mean of dataset: {}."
                  "Consider to increase the penalty!"
                  "".format(coeff[0], mean_val))

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
            min_val = np.min(pred) - 1
            max_val = np.max(pred) + 1
            ax.plot([min_val, max_val], [min_val, max_val])
            ax.plot([min_val, max_val], [min_val+rmse, max_val+rmse], "--")
            ax.plot([min_val, max_val], [min_val-rmse, max_val-rmse], "--")
            self.plot_fitting_coefficients()
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

    @staticmethod
    def order2string(order):
        """Convert an order array into string representation."""
        nparray = np.array(order)
        num_x = np.sum(nparray==0)
        num_y = np.sum(nparray==1)
        num_z = np.sum(nparray==2)
        string_repr = "$"
        if num_x == 0 and num_y == 0 and num_z == 0:
            return "constant"
        if num_x > 0:
            string_repr += "x^{}".format(num_x)
        if num_y > 0 :
            string_repr += "y^{}".format(num_y)
        if num_z > 0:
            string_repr += "z^{}".format(num_z)
        string_repr += "$"
        return string_repr

    def plot_fitting_coefficients(self):
        """Create a plot of all the fitting coefficients."""
        from matplotlib import pyplot as plt
        coeff = self.linear_fit["coeff"]
        order = self.linear_fit["order"]

        data = {}
        annotations = {}
        for c, o in zip(coeff, order):
            if len(o) == 0:
                continue
            n = len(o)
            if n not in data.keys():
                data[n] = [c]
                annotations[n] = [WulffConstruction.order2string(o)]
            else:
                data[n].append(c)
                annotations[n].append(WulffConstruction.order2string(o))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        start = 0
        keys = list(data.keys())
        keys.sort()
        for k in keys:
            x = list(range(start, start+len(data[k])))
            ax.bar(x, data[k], label=str(k))
            start += len(data[k]) + 1
            for i in range(len(data[k])):
                ax.annotate(annotations[k][i], xy=(x[i], data[k][i]))
        ax.set_ylabel("Fitting coefficient")
        ax.set_xticklabels([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(frameon=False)
        return fig


    def wulff_plot(self, show=False, n_angles=120):
        """Create a Wulff plot."""
        try:
            from matplotlib import pyplot as plt
            fig_xy = plt.figure()
            ax_xy = fig_xy.add_subplot(1, 1, 1)
            pos = self.cluster.get_positions()
            com = np.mean(pos, axis=0)
            pos -= com

            # Project atomic positions into the xy plane
            proj_xy = pos[:, :2]
            ax_xy.plot(proj_xy[:, 0], proj_xy[:, 1], 'x')
            theta = np.zeros(n_angles) + np.pi/2.0
            theta = theta.tolist()
            phi = np.linspace(0.0, 2.0*np.pi, n_angles).tolist()
            gamma = np.array([self.eval(t, p) for t, p in zip(theta, phi)])
            x = gamma * np.cos(phi)
            y = gamma * np.sin(phi)
            ax_xy.plot(x, y)
        except Exception as exc:
            print("Could not plot because of "
                  "{}: {}".format(type(exc).__name__, str(exc)))

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
            mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            mlab.mesh(X, Y, Z, scalars=Gamma/np.min(Gamma))
            mlab.colorbar()
            if show:
                mlab.show()
        except ImportError as exc:
            print("{}: {}".format(type(exc).__name__, str(exc)))
            print("To visualize in 3D mayavi is required!")

        if show:
            plt.show()
        return fig_xy

    def path_plot(self, path=[90, 90], num_points=100, normalization=1.0, latex=False):
        """Create a wulff plot along a path.
        
        :param path: Path along which ti visualize. The first angle 
            represents rotation around the y-axis. All angles are 
            given in degrees.

            Example:
            If theta is the polar angle, and phi is the azimuthal angle
            [90, 90] means the path consists of the segments
            [0, 0] --> [0, 90] --> [90, 90] --> [90, 0]
            where each tuple denots  [theta, phi] pairs
        :type path: list of ints
        :param int num_points: Number of points along each path
        :param float normalization: Normalization factor to convert
            relative surface tension into absolute.
        :param bool latex: If True axis text will be put as raw strings
        """
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_values = []
        tick_label = []

        # Plot the first path
        end = path[0]*np.pi/180.0
        theta = np.linspace(0.0, end, num_points).tolist()
        phi = 0.0
        counter = 0
        gamma = []
        angles = []
        tick_label.append(0)
        for t in theta:
            x_values.append(counter)
            gamma.append(self.eval(t, phi))
            angles.append((t, phi))
            counter += 1

        # Plot second path
        theta = end
        end = path[1]*np.pi/180.0
        phi = np.linspace(0.0, end, num_points).tolist()
        counter -= 1
        tick_label.append(counter)
        for p in phi:
            x_values.append(counter)
            gamma.append(self.eval(theta, p))
            angles.append((theta, p))
            counter += 1

        # Plot third path (back to origin)
        theta = np.linspace(0.0, theta, num_points)[::-1]
        theta = theta.tolist()
        phi = end
        counter -= 1
        tick_label.append(counter)
        for t in theta:
            x_values.append(counter)
            gamma.append(self.eval(t, phi))
            counter += 1
            angles.append((t, phi))
        tick_label.append(counter-1)

        gamma = np.array(gamma)*normalization
        ax.plot(x_values, gamma)
        if latex:
            ax.set_ylabel(r"Surface tension (mJ/\$m^2\$")
        else:
            ax.set_ylabel("Surface tension (mJ/$m^2$")
        ax.set_xticklabels([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks(tick_label)
        ax.set_xticklabels([(0, 0), (path[0], 0), (path[0], path[1]), (0, path[1])])
        return fig
        
