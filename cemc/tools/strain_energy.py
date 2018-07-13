"""Class for calculating the strain energy of ellipsoidal inclusions."""
import numpy as np
from cemc.ce_updater import EshelbyTensor, EshelbySphere


class StrainEnergy(object):
    """Class for calculating strain energy of ellipsoidal inclusions."""

    def __init__(self, aspect=[1.0, 1.0, 1.0],
                 eigenstrain=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], poisson=0.3):
        """Initialize Strain energy class."""
        aspect = np.array(aspect)
        tol = 1E-6
        if np.all(np.abs(aspect-aspect[0]) < tol):
            self.eshelby = EshelbySphere(aspect[0], poisson)
        else:
            self.eshelby = EshelbyTensor(aspect[0], aspect[1], aspect[2],
                                         poisson)
        self.eigenstrain = np.array(eigenstrain)

    def equivalent_eigenstrain(self, scale_factor):
        """Compute the equivalent eigenstrain.

        :param elast_matrix: Elastic tensor of the matrix material
        :param scale_factor: The elastic tensor of the inclustion is assumed to
                             be scale_factor*elast_matrix
        """
        S = np.array(self.eshelby.aslist())
        A = (scale_factor-1.0)*S + np.identity(6)
        return np.linalg.solve(A, scale_factor*self.eigenstrain)
