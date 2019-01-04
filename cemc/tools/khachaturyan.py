import numpy as np
from cemc.tools import to_full_rank4
from itertools import product

class Khachaturyan(object):
    def __init__(self, elastic_tensor=None, uniform_strain=None, 
                 misfit_strain=None):
        self.C = to_full_rank4(elastic_tensor)
        self.uniform_strain = uniform_strain
        if self.uniform_strain is None:
            self.uniform_strain = np.zeros((3, 3))

        if misfit_strain is None:
            raise ValueError("Misfit strain has to be a 3x3 numpy array!")
        self.misfit_strain = misfit_strain

    def zeroth_order_green_function(self, nhat):
        """Calculate the zeroth order Green function (Fourier representation).
           The prefactor 1/k^2 is omitted.

        :param np.ndarray nhat: Unit vector in the reciprocal direction
        """
        Q = np.einsum("m,n,lmnp->lp", nhat, nhat, self.C)
        return np.linalg.inv(Q)

    def effective_stress(self):
        """Calculate the stress resulting from the misfit strain.
           This only to zeroth order (i.e. effects of different elastic
           constants of the inclusion and the homogeneous strain is 
           not included)
        """
        return np.einsum("ijkl,kl", self.C, self.misfit_strain)

    def strain_energy_voxels(self, shape_function):
        """Calculate the strain energy for the given shape function."""
        V = np.sum(shape_function)
        ft = np.abs(np.fft.fftn(shape_function))**2
        ft /= np.prod(ft.shape)
        freqs = []
        for i in range(len(shape_function.shape)):
            freqs.append(np.fft.fftfreq(shape_function.shape[i]))
        eff = self.effective_stress()

        indices = [range(len(f)) for f in freqs]
        for indx in product(*indices):
            k = np.array([freqs[i][indx[i]] for i in range(len(freqs))])
            if np.allclose(k, 0.0):
                continue
            khat = k/np.sqrt(k.dot(k))
            G = self.zeroth_order_green_function(khat)
            val = np.einsum("ik,k,ij,jl,l", eff, khat, G, eff, khat)
            ft[indx[0], indx[1], indx[2]] *= val
        
        diff = self.misfit_strain - self.uniform_strain
        energy = 0.5*np.einsum("ijkl,ij,kl", self.C, diff, diff)
        return energy - 0.5*np.sum(ft)/V

    