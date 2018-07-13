"""Class for calculating the strain energy of ellipsoidal inclusions."""
import numpy as np
from cemc.ce_updater import EshelbyTensor, EshelbySphere
from cemc.tools import rot_matrix, rotate_tensor, to_voigt, to_full_tensor
from itertools import product


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
            eshelby = EshelbySphere(aspect[0], poisson)
        else:
            eshelby = EshelbyTensor(aspect[0], aspect[1], aspect[2],
                                    poisson)
        return eshelby

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

    def explore_orientations(self, aspect, scale_factor, e_matrix, step=10):
        """Explore the strain energy as a function of ellipse orientation."""

        aspect = np.array(aspect)
        self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
        result = []
        eigenstrain_orig = to_full_tensor(self.eigenstrain)
        for euler in product(np.arange(0, 360, step), repeat=3):
            sequence = [("x", euler[0]), ("z", euler[1]), ("x", euler[2])]
            matrix = rot_matrix(sequence)
            strain = rotate_tensor(eigenstrain_orig, matrix)
            self.eigenstrain = to_voigt(strain)
            energy = self.strain_energy(scale_factor, e_matrix)
            result.append({"energy": energy, "rot_seq": sequence})
        self.summarize_orientation_serch(result)
        return result

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
            self.log("{:3} \t {}".format(top_20[i]["energy"],
                                         top_20[i]["rot_seq"]))
        self.log("---------------------------------------------------------")

    @staticmethod
    def volume_ellipsoid(aspect):
        """Compute the volume of an ellipsoid."""
        return 4.0*np.pi*aspect[0]*aspect[1]*aspect[2]/3.0

    def plot(self, scale_factor, elast_matrix, log=True):
        """Create a plot of the energy as different aspect ratios."""
        from matplotlib import pyplot as plt

        a_over_c = np.logspace(0, 3, 100)
        b_over_c = [1, 10, 100]

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
                E = self.strain_energy(scale_factor, elast_matrix)
                W.append(E)
                a_plot.append(a)
            ax.plot(a_plot, W, label="{}".format(int(b)))

        # Separation line
        b_sep = np.logspace(0, 3, 100)
        W = []
        for b in b_sep:
            aspect = [b, b, 1.0]
            self.eshelby = StrainEnergy.get_eshelby(aspect, self.poisson)
            E = self.strain_energy(scale_factor, elast_matrix)
            W.append(E)

        ax.plot(b_sep, W, "--", color="grey")
        ax.legend()
        if log:
            ax.set_xscale("log")
        return fig
