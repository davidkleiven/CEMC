import numpy as np
from ase.geometry import wrap_positions

class MultivariateGaussian(object):
    """
    Class for a periodically repeated Gaussian distribution

    :param numpy.ndarray cell: 3x3 matrix representing the cell
        each row is a lattice vector
    :param numpy.ndarray mu: Vector of length 3 representing the mean
    :param numpy.ndarray sigma: Covariance matrix (3x3)
    """
    def __init__(self, mu=np.zeros(3), sigma=np.eye(3)):
        self.mu = mu
        self._sigma = sigma
        self.normalization = 1.0/np.sqrt((2.0*np.pi)**3 *np.linalg.det(self._sigma))
        self.inv_sigma = np.linalg.inv(self._sigma)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = new_sigma
        self.inv_sigma = np.linalg.inv(self._sigma)

        self.normalization = 1.0/np.sqrt((2.0*np.pi)**3 * np.linalg.det(self._sigma))

    def __call__(self, x):
        values = x - self.mu
        weight = np.exp(-0.5*values.dot(self.inv_sigma.dot(values.T)))
        return weight * self.normalization

