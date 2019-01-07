import numpy as np
from cemc.tools import to_full_rank4
from itertools import product
from cemc_cpp_code import PyKhachaturyan
from cemc.tools import rotate_tensor, rot_matrix, rotate_rank4_tensor
import time
import datetime

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

    def zeroth_order_integral_pure_python(self, ft):
        freqs = []
        for i in range(len(ft.shape)):
            freqs.append(np.fft.fftfreq(ft.shape[i]))
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
        return np.sum(ft)

    def strain_energy_voxels(self, shape_function, pure_python=False):
        """Calculate the strain energy for the given shape function."""
        V = np.sum(shape_function)
        ft = np.abs(np.fft.fftn(shape_function))**2
        ft /= np.prod(ft.shape)

        if pure_python:
            integral = self.zeroth_order_integral_pure_python(ft)
        else:
            pykhach = PyKhachaturyan(ft, self.C, self.misfit_strain)
            integral = pykhach.zeroth_order_integral()

        diff = self.misfit_strain - self.uniform_strain
        energy = 0.5*np.einsum("ijkl,ij,kl", self.C, diff, diff)
        return energy - 0.5*integral/V

    def explore_orientations(self, voxels, theta_ax="y", phi_ax="z", step=5,
                             theta_min=0, theta_max=180, phi_min=0, phi_max=360,
                             fname=None):
        """Explore orientation dependency of the strain energy.
        
        :param np.ndarray voxels: Voxel representation of the geometry
        :param str theta_ax: Rotation axis for theta angle
        :param str phi_ax: Rotation axis for phi angle
        :param int step: Angle change in degress
        :param int theta_min: Start angle for theta
        :param int theta_max: End angle for theta
        :param int phi_min: Start angle for phi
        :param int phi_max: End angle for phi
        :param fname str: Filename for storing the output result
        """
        th = list(range(theta_min, theta_max, step))
        ph = list(range(phi_min, phi_max, step))

        misfit_orig = self.misfit_strain.copy()
        orig_C = self.C.copy()
        result = []
        now = time.time()
        status_interval = 30
        for ang in product(th, ph):
            if time.time() - now > status_interval:
                print("Theta: {}, Phi: {}".format(ang[0], ang[1]))
                now = time.time()
            seq = [(theta_ax, -ang[0]), (phi_ax, -ang[1])]
            matrix = rot_matrix(seq)

            # Rotate the strain tensor
            self.misfit_strain = rotate_tensor(misfit_orig, matrix)
            self.C = rotate_rank4_tensor(orig_C.copy(), matrix)
            energy = self.strain_energy_voxels(voxels)
            result.append([ang[0], ang[1], energy])

        if fname is None:
            fname = "khacaturyan_orientaions{}.csv".format(timestamp)
        
        np.savetxt(fname, np.array(result), delimiter=",", header=
                   "Theta ({}) deg, Phi ({}) deg, Energy (eV)".format(theta_ax, phi_ax))
        print("Results of orientation exploration written to {}".format(fname))

def timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')





    