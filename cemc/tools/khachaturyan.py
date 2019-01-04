import numpy as np
from cemc.tools import to_full_rank4

class Khachaturyan(object):
    def __init__(self, elastic_tensor=None):
        self.C = elastic_tensor

    def zeroth_order_green_function(self, nhat):
        """Calculate the zeroth order Green function (Fourier representation).
           The prefactor 1/k^2 is omitted.

        :param np.ndarray nhat: Unit vector in the reciprocal direction
        """
        full_tensor = to_full_rank4(self.C)
        Q = np.einsum("m,n,lmnp->lp", nhat, nhat, full_tensor)
        return np.linalg.inv(Q)