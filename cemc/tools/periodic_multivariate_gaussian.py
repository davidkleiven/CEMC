import numpy as np
from ase.geometry import wrap_positions

class PeriodicMultivariateGaussian(object):
    """
    Class for a periodically repeated Gaussian distribution

    :param numpy.ndarray cell: 3x3 matrix representing the cell
        each row is a lattice vector
    :param numpy.ndarray mu: Vector of length 3 representing the mean
    :param numpy.ndarray sigma: Covariance matrix (3x3)
    """
    def __init__(self, cell=None, mu=np.zeros(3), sigma=np.eye(3)):
        self._mu = mu
        self._sigma = sigma
        self.cell = cell
        self.centers = self._get_centers()
        self.normalization = 1.0/np.sqrt(2.0*np.pi*np.linalg.det(self._sigma))
        self.inv_sigma = np.linalg.inv(self._sigma)
        
        if self.cell is None:
            raise TypeError("Cell has to be given!")

    def _get_centers(self):
        """Calculate all periodic images."""
        from itertools import product
        centers = np.zeros((27, 3))
        for i, comb in enumerate(product([-1, 0, 1], repeat=3)):
            vec = self.cell.T.dot(comb)
            centers[i, :] = vec
        return centers

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = new_sigma
        self.inv_sigma = np.linalg.inv(self._sigma)
        self.normalization = 1.0/np.sqrt(2.0*np.pi*np.linalg.det(self._sigma))

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, new_val):
        self._mu = wrap_positions(self.cell, new_val)

    def __call__(self, x):
        values = x - (self.centers + self._mu)
        weight = np.mean(np.exp(-0.5*values.dot(self.inv_sigma.dot(values.T))))
        return weight/self.normalization

