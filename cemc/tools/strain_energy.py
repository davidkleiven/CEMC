"""Class for calculating the strain energy of ellipsoidal inclusions."""
import numpy as np
from cemc_cpp_code import PyEshelbyTensor, PyEshelbySphere
from cemc.tools import rot_matrix, rotate_tensor, to_mandel, to_full_tensor
from cemc.tools import rot_matrix_spherical_coordinates
from cemc.tools import rotate_rank4_mandel
from itertools import product
from scipy.optimize import minimize


class StrainEnergy(object):
    """Class for calculating strain energy of ellipsoidal inclusions.
    
    :param list aspect: Aspect ratio of the ellipsoid. 
        NOTE: The convention aspect[0] >= aspect[1] >= aspect[2]
        is used. If the ellipsoid is oriented in a different way,
        it has to be rotated after.
    :param misfit: Misfit strain of the inclusion
    :type misfit: 3x3 ndarray or list of length 6 (Mandel notation)
    :param float poisson: Poisson ratio
    """

    def __init__(self, aspect=[1.0, 1.0, 1.0],
                 misfit=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], poisson=0.3):
        """Initialize Strain energy class."""
        aspect = np.array(aspect)
        self.eshelby = StrainEnergy.get_eshelby(aspect, poisson)
        self.misfit = np.array(misfit)

        if len(self.misfit.shape) == 2:
            self.misfit = to_mandel(self.misfit)
        self.poisson = poisson

    @staticmethod
    def get_eshelby(aspect, poisson):
        """Return the Eshelby tensor.
        
        :param float poisson: Poisson ratio
        """
        tol = 1E-6
        if np.all(np.abs(aspect-aspect[0]) < tol):
            eshelby = PyEshelbySphere(aspect[0], aspect[0], aspect[0], poisson)
        else:
            eshelby = PyEshelbyTensor(aspect[0], aspect[1], aspect[2],
                                    poisson)
        return eshelby

    def _check_ellipsoid(self, ellipsoid):
        """Check that the ellipsoid arguments is correct.
        
        :param dict ellipsoid: Dictionary describing the ellipsoid
        """
        required_keys = ["aspect"]
        for key in ellipsoid.keys():
            if key not in required_keys:
                msg = "The ellipsoid dictionary has to "
                msg += "include {}".format(required_keys)
                raise ValueError(msg)

        if len(ellipsoid["aspect"]) != 3:
            raise ValueError("aspect ratio should be a list/array of length 3")

    def equivalent_eigenstrain(self, C_matrix=None, C_prec=None, 
                               scale_factor=None):
        """Compute the equivalent eigenstrain.

        :param ndarray C_matrix: 6x6 elastic tensor of the matrix material
        :param ndarray C_prec: 6x6 elastic tensor of the inclusion
        :param float scale_factor: The elastic tensor of the inclustion is assumed to
                             be scale_factor*elast_matrix
        """
        if C_matrix is None:
            raise ValueError("Elastic tensor for the matrix material "
                             "must be passed!")

        if C_prec is None and scale_factor is not None:
            C_prec = scale_factor*C_matrix

        if C_prec is None:
            raise ValueError("Elastic tensor or a scale factor for "
                             "the precipitating material must be "
                             "passed")
        S = np.array(self.eshelby.aslist())
        A = (C_prec - C_matrix).dot(S) + C_matrix
        b = C_prec.dot(self.misfit)
        return np.linalg.solve(A, b)

    def stress(self, equiv_strain, C_matrix=None):
        """Compute the stress tensor.

        :param list equiv_strain: Equivalent eigenstrain in Mandel notation
        :param C_matrix: 6x6 elastic tensor of the matrix material (Mandel)
        """
        S = np.array(self.eshelby.aslist())
        sigma = C_matrix.dot(S.dot(equiv_strain) - equiv_strain)
        return sigma

    def strain_energy(self, C_matrix=None, C_prec=None, 
                      scale_factor=None):
        """Compute the strain energy per volume.
        
        :param ndarray C_matrix: 6x6 elastic tensor of the matrix material
        :param ndarray C_prec: 6x6 elastic tensor of the precipitate material
        :param float scale_factor: If given and C_prec=None, 
            C_pref = scale_factor*C_matrix
        """
        eq_strain = self.equivalent_eigenstrain(
            C_matrix=C_matrix, C_prec=C_prec, 
            scale_factor=scale_factor)
        sigma = self.stress(eq_strain, C_matrix)

        # Off diagonal elements should be multiplied by sqrt(2)
        strain = self.misfit.copy()
        return -0.5*sigma.dot(strain)

    def is_isotropic(self, matrix, mat_type="mandel"):
        """Check tensor represent an isotropic material."""
        factor = 1.0
        if mat_type == "mandel":
            factor = 2.0
        shear = matrix[3, 3]/factor
        if not np.allclose(np.diag(matrix)[3:]/factor, shear):
            return False

        if not np.allclose(np.diag(matrix)[:3], matrix[0, 0]):
            return False

        if not np.allclose(matrix[:3, 3:], 0.0):
            return False
        
        if not np.allclose(matrix[3:, :3], 0.0):
            return False
        
        # Check that all off diagonal elements in the uppder
        # 3x3 are the same
        for indx in product([0, 1, 2], repeat=2):
            if indx[0] == indx[1]:
                continue
            if abs(matrix[indx[0], indx[1]] - matrix[0, 1]) > 1E-4:
                return False

        # At this point we know that the material is
        # isotropic. Just apply one final consistency check
        # bulk_mod = matrix[0, 0] - 4.0*shear/3.0
        # expected = bulk_mod - 2.0*shear/3.0
        # print(matrix[0,1 ], factor*expected)
        # assert abs(matrix[0, 1] - factor*expected) < 1E-6
        return True

    def explore_orientations(self, ellipsoid, C_matrix, step=10,
                             fname="", theta_ax="y", phi_ax="z"):
        """Explore the strain energy as a function of ellipse orientation.
        
        :param dict ellipsoid: Dictionary with information of the ellipsoid
            The format should be {"aspect": [1.0, 1.0, 1.0], "C_prec": ..., 
            "scale_factor": 1.0}
            C_prec is the elastic tensor of the precipitate material.
            If not given, scale_factor has to be given, and the elastic
            tensor of the precipitate material is taken as this factor
            multiplied by the elastic tensor of the matrix material
        :param numpy.ndarray C_matrix: Elastic tensor of the matrix material
        :param float step: Angle step size in degree
        :param str fname: If given the result of the exploration is 
            stored in a csv file with this filename
        :param str theta_ax: The first rotation is performed
            around this axis
        :param str phi_ax: The second rotation is performed around this
            axis (in the new coordinate system after the first rotation
            is performed)
        """
        from itertools import product
        #self._check_ellipsoid(ellipsoid)
        scale_factor = ellipsoid.get("scale_factor", None)
        C_prec = ellipsoid.get("C_prec", None)

        if C_prec is None:
            C_prec = scale_factor*C_matrix
        aspect = np.array(ellipsoid["aspect"])
        self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
        result = []
        misfit_orig = to_full_tensor(self.misfit)
        theta = np.arange(0.0, np.pi, step*np.pi / 180.0)
        phi = np.arange(0.0, 2.0 * np.pi, step * np.pi / 180.0)
        theta = np.append(theta, [np.pi])
        phi = np.append(phi, [2.0*np.pi])

        C_matrix_orig = C_matrix.copy()
        C_prec_orig = C_prec.copy()
        for ang in product(theta, phi):
            th = ang[0]
            p = ang[1]
            theta_deg = th*180/np.pi
            phi_deg = p*180/np.pi
            seq = [(theta_ax, -theta_deg), (phi_ax, -phi_deg)]
            matrix = rot_matrix(seq)
            #matrix = rot_matrix_spherical_coordinates(p, th)

            # Rotate the strain tensor
            strain = rotate_tensor(misfit_orig, matrix)
            self.misfit = to_mandel(strain)


            # Rotate the elastic tensor of the matrix material
            C_matrix = rotate_rank4_mandel(C_matrix_orig, matrix)

            # Rotate the elastic tensor of the precipitate material
            C_prec = rotate_rank4_mandel(C_prec_orig, matrix)            
            if abs(p) < 1E-3 and (abs(th-np.pi/4.0) < 1E-3 or abs(th-3.0*np.pi/4.0) < 1E-3):
                print(self.eshelby.aslist())

            energy = self.strain_energy(C_matrix=C_matrix, C_prec=C_prec)
            res = {"energy": energy, "theta": th, "phi": p}
            a = matrix.T.dot([1.0, 0.0, 0.0])
            b = matrix.T.dot([0.0, 1.0, 0.0])
            c = matrix.T.dot([0.0, 0.0, 1.0])
            res["half_axes"] = {"a": a, "b": b, "c": c}
            res["misfit"] = self.misfit
            result.append(res)

        if fname != "":
            self.save_orientation_result(result, fname)

        # Sort the result from low energy to high energy
        energies = [res["energy"] for res in result]
        sorted_indx = np.argsort(energies)
        result = [result[indx] for indx in sorted_indx]

        # Reset the strain
        self.misfit = to_mandel(misfit_orig)
        return result

    def save_orientation_result(self, result, fname):
        """Store the orientation result.
        
        :param list result: List with result from exploration
            each item is a dictionary with the containing keys
            theta, phi and energy.
        :param str fname: Filename (csv-file)
        """
        theta = [res["theta"] for res in result]
        phi = [res["phi"] for res in result]
        energy = [res["energy"] for res in result]

        data = np.vstack((theta, phi, energy)).T
        np.savetxt(fname, data, header="Polar angle, Azm. angle, Energy",
                   delimiter=",")
        print("Orientation results written to {}".format(fname))

    def log(self, msg):
        """Log message to screen."""
        print(msg)

    def plot_explore_result(self, explore_result, latex=False):
        """Plot a diagonistic plot over the exploration result."""
        from matplotlib import pyplot as plt
        from scipy.interpolate import SmoothSphereBivariateSpline
        from scipy.interpolate import griddata
        energy = []
        phi = []
        theta = []
        for res in explore_result:
            energy.append(res["energy"]*1000.0)
            phi.append(res["phi"])
            theta.append(res["theta"])
        
        th_fine = np.linspace(0.0, np.pi, 90)
        phi_fine = np.linspace(0.0, 2.0*np.pi, 90)
        phi_min = np.min(phi_fine) * 180.0 / np.pi
        phi_max = np.max(phi_fine) * 180.0 / np.pi
        theta_min = np.min(th_fine) * 180.0 / np.pi
        theta_max = np.max(th_fine) * 180.0 / np.pi

        # Create plot with griddata
        data = np.vstack((theta, phi)).T
        T, P = np.meshgrid(th_fine, phi_fine)
        energy_interp = griddata(data, energy, (T, P))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(energy_interp, cmap="inferno",
                       extent=[theta_min, theta_max, phi_min, phi_max],
                       aspect="auto", origin="lower")
        ax.set_xlabel("Polar angle (deg)")
        ax.set_ylabel("Azimuthal angle (deg)")
        cbar = fig.colorbar(im)
        cbar.set_label("Strain energy (meV per angstrom cubed)")
        return fig

    @staticmethod
    def volume_ellipsoid(aspect):
        """Compute the volume of an ellipsoid."""
        return 4.0*np.pi*aspect[0]*aspect[1]*aspect[2]/3.0


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
