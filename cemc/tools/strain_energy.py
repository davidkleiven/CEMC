"""Class for calculating the strain energy of ellipsoidal inclusions."""
import numpy as np
from cemc_cpp_code import PyEshelbyTensor, PyEshelbySphere
from cemc.tools import rot_matrix, rotate_tensor, to_voigt, to_full_tensor
from cemc.tools import rot_matrix_spherical_coordinates
from itertools import product
from scipy.optimize import minimize


class StrainEnergy(object):
    """Class for calculating strain energy of ellipsoidal inclusions."""

    def __init__(self, aspect=[1.0, 1.0, 1.0],
                 eigenstrain=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], poisson=0.3):
        """Initialize Strain energy class."""
        aspect = np.array(aspect)
        self.eshelby = StrainEnergy.get_eshelby(aspect, poisson)
        self.eigenstrain = np.array(eigenstrain)
        self.poisson = poisson

    @staticmethod
    def get_eshelby(aspect, poisson):
        """Return the Eshelby tensor."""
        tol = 1E-6
        if np.all(np.abs(aspect-aspect[0]) < tol):
            eshelby = PyEshelbySphere(aspect[0], aspect[0], aspect[0], poisson)
        else:
            eshelby = PyEshelbyTensor(aspect[0], aspect[1], aspect[2],
                                    poisson)
        return eshelby

    def _check_ellipsoid(self, ellipsoid):
        """Check that the ellipsoid arguments is correct."""
        required_keys = ["aspect", "scale_factor"]
        for key in ellipsoid.keys():
            if key not in required_keys:
                msg = "The ellipsoid dictionary has to"
                msg += "include {}".format(required_keys)
                raise ValueError(msg)

        if len(ellipsoid["aspect"]) != 3:
            raise ValueError("aspect ratio should be a list/array of length 3")

    def equivalent_eigenstrain(self, scale_factor):
        """Compute the equivalent eigenstrain.

        :param elast_matrix: Elastic tensor of the matrix material
        :param scale_factor: The elastic tensor of the inclustion is assumed to
                             be scale_factor*elast_matrix
        """
        S = np.array(self.eshelby.aslist())
        A = (scale_factor-1.0)*S + np.identity(6)
        return np.linalg.solve(A, scale_factor*self.eigenstrain)

    def stress(self, equiv_strain, elastic_matrix):
        """Compute the stress tensor.

        :param equiv_strain: Equivalent eigenstrain
        :param elastic_matrix: Elastic tensor of the matrix material
        """
        S = np.array(self.eshelby.aslist())
        sigma = elastic_matrix.dot(S.dot(equiv_strain) - equiv_strain)
        return sigma

    def strain_energy(self, scale_factor, elast_matrix):
        """Compute the strain energy per volume."""
        eq_strain = self.equivalent_eigenstrain(scale_factor)
        sigma = self.stress(eq_strain, elast_matrix)

        # Off diagonal elements should be multiplied by sqrt(2)
        strain = self.eigenstrain.copy()
        strain[3:] = np.sqrt(2)*strain[3:]
        sigma[3:] = np.sqrt(2)*sigma[3:]
        return -0.5*sigma.dot(strain)

    def explore_aspect_ratios(self, scale_factor, e_matrix,
                              angle_step=30):
        """
        Exploring aspect ratios for needle, plate and spheres
        """

        ellipsoids = {
            "plate": {
                        "scale_factor": scale_factor,
                        "aspect": [1000.0, 1000.0, 1.0]
                    },
            "sphere": {
                        "scale_factor": scale_factor,
                        "aspect": [1.0, 1.0, 1.0]
                    },
            "needle": {
                        "scale_factor": scale_factor,
                        "aspect": [1000.0, 1.0, 1.0]
                    }
        }
        self.log("\n")
        for key, value in ellipsoids.items():
            res = self.explore_orientations(value, e_matrix, step=angle_step)

            self.log("Minmum energy for {} inclusion:".format(key))
            E = res[0]["energy"]*1000.0
            rot = res[0]["rot_seq"]
            self.log("Energy: {} meV/A^3. Rotation: {}".format(E, rot))
            self.log("\n")

    def explore_orientations(self, ellipsoid, e_matrix, step=10,
                             print_summary=False):
        """Explore the strain energy as a function of ellipse orientation."""
        self._check_ellipsoid(ellipsoid)
        scale_factor = ellipsoid["scale_factor"]
        aspect = np.array(ellipsoid["aspect"])
        self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
        result = []
        eigenstrain_orig = to_full_tensor(self.eigenstrain)
        theta = np.arange(0.0, np.pi, step*np.pi / 180.0)
        phi = np.arange(0.0, 2.0 * np.pi, step * np.pi / 180.0)

        for th in theta:
            for p in phi:
                matrix = rot_matrix_spherical_coordinates(p, th)
                strain = rotate_tensor(eigenstrain_orig, matrix)
                self.eigenstrain = to_voigt(strain)
                energy = self.strain_energy(scale_factor, e_matrix)
                res = {"energy": energy, "theta": th, "phi": p}
                a = matrix.T.dot([1.0, 0.0, 0.0])
                b = matrix.T.dot([0.0, 1.0, 0.0])
                c = matrix.T.dot([0.0, 0.0, 1.0])
                res["half_axes"] = {"a": a, "b": b, "c": c}
                res["eigenstrain"] = self.eigenstrain
                result.append(res)

        if print_summary:
            self.summarize_orientation_serch(result)

        # Sort the result from low energy to high energy
        energies = [res["energy"] for res in result]
        sorted_indx = np.argsort(energies)
        result = [result[indx] for indx in sorted_indx]

        # Reset the strain
        self.eigenstrain = to_voigt(eigenstrain_orig)
        return result

    def optimize_rotation(self, ellipsoid, e_matrix, init_rot):
        """Optimize a rotation."""
        self._check_ellipsoid(ellipsoid)
        axes, angles = unwrap_euler_angles(init_rot)
        orig_strain = to_full_tensor(self.eigenstrain.copy())
        aspect = np.array(ellipsoid["aspect"])

        self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
        opt_args = {
            "ellipsoid": ellipsoid,
            "elast_matrix": e_matrix,
            "strain_energy_obj": self,
            "orig_strain": orig_strain,
            "rot_axes": axes
        }
        opts = {"eps": 1.0}
        res = minimize(cost_minimize_strain_energy, angles, args=(opt_args,),
                       options=opts)
        rot_seq = combine_axes_and_angles(axes, res["x"])
        optimal_orientation = {
            "energy": res["fun"],
            "rot_sequence": rot_seq,
            "rot_matrix": rot_matrix(rot_seq)
        }

        # Reset the eigenstrain back
        self.eigenstrain = to_voigt(orig_strain)
        return optimal_orientation

    def log(self, msg):
        """Log message to screen."""
        print(msg)

    def summarize_orientation_serch(self, result):
        """Prints a summary of the orientation search."""
        num_angles = 20
        self.log("=========================================================")
        self.log("==            SUMMARY FROM ORIETATION SEARCH           ==")
        self.log("=========================================================")
        energies = [res["energy"] for res in result]
        max = np.max(energies)
        min = np.min(energies)
        indices = np.argsort(energies)[:num_angles]
        top_20 = [result[indx] for indx in indices]

        self.log("Maximum energy: {}".format(max))
        self.log("Minumum energy: {}".format(min))
        self.log("{} orientation with the lowest energy".format(num_angles))
        self.log("---------------------------------------------------------")
        for i in range(num_angles):
            out = "{:3} \t {} \t {}".format(top_20[i]["energy"]*1000.0,
                                            top_20[i]["theta"], top_20[i]["phi"])
            a = np.round(top_20[i]["half_axes"]["a"], decimals=2)
            b = np.round(top_20[i]["half_axes"]["b"], decimals=2)
            c = np.round(top_20[i]["half_axes"]["c"], decimals=2)
            out += "a: {} \t b: {} c: {}\t".format(a, b, c)
            out += "{}".format(top_20[i]["eigenstrain"])
            self.log(out)
        self.log("---------------------------------------------------------")

    def plot_explore_result(self, explore_result, latex=False):
        """Plot a diagonistic plot over the exploration result."""
        from matplotlib import pyplot as plt
        from scipy.interpolate import SmoothSphereBivariateSpline
        energy = []
        phi = []
        theta = []
        for res in explore_result:
            energy.append(res["energy"]*1000.0)
            phi.append(res["phi"])
            theta.append(res["theta"])

        lut = SmoothSphereBivariateSpline(theta, phi, energy, s=3.5)
        th_fine = np.linspace(0.0, np.pi, 90)
        phi_fine = np.linspace(0.0, 2.0*np.pi, 90)
        data_smooth = lut(th_fine, phi_fine)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        phi_min = np.min(phi_fine) * 180.0 / np.pi
        phi_max = np.max(phi_fine) * 180.0 / np.pi
        theta_min = np.min(th_fine) * 180.0 / np.pi
        theta_max = np.max(th_fine) * 180.0 / np.pi
        im = ax.imshow(data_smooth, cmap="inferno",
                       extent=[theta_min, theta_max, phi_min, phi_max],
                       aspect="auto", origin="lower")
        cb = fig.colorbar(im)
        cb.set_label("Strain energy (meV per angstrom cubed)")
        ax.set_xlabel("Polar angle (deg)")
        ax.set_ylabel("Azimuthal angle (deg)")
        return fig

    @staticmethod
    def volume_ellipsoid(aspect):
        """Compute the volume of an ellipsoid."""
        return 4.0*np.pi*aspect[0]*aspect[1]*aspect[2]/3.0

    def plot(self, scale_factor, elast_matrix, rot_seq=None, latex=False):
        """Create a plot of the energy as different aspect ratios."""
        from matplotlib import pyplot as plt
        a_over_c = np.logspace(0, 3, 100)
        b_over_c = [1, 2, 5, 10, 50, 100]

        orig_strain = self.eigenstrain.copy()

        if rot_seq is not None:
            strain = to_full_tensor(orig_strain)
            rot_mat = rot_matrix(rot_seq)
            strain = rotate_tensor(strain, rot_mat)
            self.eigenstrain = to_voigt(strain)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for b in b_over_c:
            W = []
            a_plot = []
            for a in a_over_c:
                if a < b:
                    continue
                aspect = [a, b, 1.0]
                self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
                E = self.strain_energy(scale_factor, elast_matrix)*1000.0
                W.append(E)
                a_plot.append(a)
            ax.plot(a_plot, W, label="{}".format(int(b)))

        # Separation line
        b_sep = np.logspace(0, 3, 100)
        W = []
        for b in b_sep:
            aspect = [b, b, 1.0]
            self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
            E = self.strain_energy(scale_factor, elast_matrix)*1000.0
            W.append(E)

        ax.plot(b_sep, W, "--", color="grey")
        ax.legend(frameon=False, loc="best")
        if latex:
            xlab = "\$a/c\$"
            ylab = "Strain energy (meV/\$\SI{}{\\angstrom^3}\$)"
        else:
            xlab = "$a/c$"
            ylab = "Strain enregy (meV/A^3)"
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xscale("log")
        self.eigenstrain = orig_strain
        return fig

    def show_ellipsoid(self, ellipsoid, rot_seq):
        """Show the ellipsoid at given orientation."""
        from matplotlib import pyplot as plt
        matrix = rot_matrix(rot_seq)
        coefs = np.array(ellipsoid["aspect"])

        # Spherical angles
        u = np.linspace(0, 2.0*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        rx, ry, rz = coefs
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))

        N_pos = x.shape[0]*x.shape[1]
        # Pack x, y, z into a numpy matrix
        all_coords = np.zeros((3, N_pos))
        all_coords[0, :] = np.ravel(x)
        all_coords[1, :] = np.ravel(y)
        all_coords[2, :] = np.ravel(z)

        # Rotate all. Use the transpose of the rotation matrix because
        # the rotation matrix is intended to be used for rotating the
        # coordinate system, keeping the ellipsoid fixed
        # so the inverse rotation is required when rotating the ellipsoid
        all_coords = matrix.T.dot(all_coords)
        x = np.reshape(all_coords[0, :], x.shape)
        y = np.reshape(all_coords[1, :], y.shape)
        z = np.reshape(all_coords[2, :], z.shape)

        # Adjustment of the axes, so that they all have the same span:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color="grey")

        max_radius = max(rx, ry, rz)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        return fig


def unwrap_euler_angles(sequence):
    """Unwrap the list of axes and angles to separate lists."""
    axes = []
    angles = []
    for item in sequence:
        axes.append(item[0])
        angles.append(item[1])
    return axes, angles


def combine_axes_and_angles(axes, angles):
    """Combine the list of axes and angles into a list of dict."""
    return zip(axes, angles)


def cost_minimize_strain_energy(euler, args):
    """Cost function used to minimize."""

    ellipsoid = args["ellipsoid"]
    scale_factor = ellipsoid["scale_factor"]
    e_matrix = args["elast_matrix"]
    orig_strain = args["orig_strain"]
    obj = args["strain_energy_obj"]
    rot_axes = args["rot_axes"]

    rot_seq = combine_axes_and_angles(rot_axes, euler)
    matrix = rot_matrix(rot_seq)
    tensor = rotate_tensor(orig_strain, matrix)
    tensor = to_voigt(tensor)
    obj.eigenstrain = tensor
    return obj.strain_energy(scale_factor, e_matrix)


def map_euler_angles_to_unique_float(euler):
    """Map the euler angles into one unique float."""
    return euler[0] + euler[1]*360 + euler[2]*360*360


def vec2polar_angles(vec):
    """Find the polar angles describing the direction of a vector."""
    vec /= np.sqrt(vec.dot(vec))
    theta = np.arccos(vec[2])
    if theta < 1E-6 or theta > np.pi-1E-6:
        phi = 0.0
    else:
        phi = np.arcsin(vec[1]/np.sin(theta))
        # assert abs(abs(vec[0]) - abs(np.cos(phi)*np.sin(theta))) < 1E-6
    return (phi, theta)
