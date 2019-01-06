import unittest
import numpy as np
from itertools import product

try:
    from cemc.tools import Khachaturyan
    from cemc_cpp_code import PyKhachaturyan
    from cemc.tools import to_full_rank4
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(reason)

class TestKhacaturyan(unittest.TestCase):
    K = 50.0
    G = 26.0

    def get_isotropic_tensor(self):
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = self.K + 4*self.G/3.0
        C[0, 1] = C[0, 2] = \
        C[1, 0] = C[1, 2] = \
        C[2, 0] = C[2, 1] = self.K - 2.0*self.G/3.0
        C[3, 3] = C[4, 4] = C[5, 5] = 2*self.G
        return C

    @property
    def poisson(self):
        return 0.5*(3*self.K - 2*self.G)/(3*self.K + self.G)

    def isotropic_green_function(self, k):
        return np.eye(3)/self.G - 0.5*np.outer(k, k)/(self.G*(1.0 - self.poisson))

    def get_sphere_voxels(self, N):
        shape_func = np.zeros((N, N, N), dtype=np.uint8)
        indx = np.array(range(N))
        ix, iy, iz = np.meshgrid(indx, indx, indx)
        r_sq = (ix-N/2)**2 + (iy-N/2)**2 + (iz-N/2)**2
        r = N/8.0
        shape_func[r_sq<r] = 1
        return shape_func

    def get_plate_voxels(self, N):
        shape_func = np.zeros((N, N, N), dtype=np.uint8)
        width = int(N/4)
        shape_func[:width, :width, :2] = 1
        return shape_func

    def get_needle_voxels(self, N):
        shape_func = np.zeros((N, N, N), dtype=np.uint8)
        width = int(N/4)
        shape_func[:width, :2, :2] = 1
        return shape_func

    def test_isotropic(self):
        if not available:
            self.skipTest(reason)
        
        misfit = np.eye(3)*0.05
        strain = Khachaturyan(elastic_tensor=self.get_isotropic_tensor(),
                              misfit_strain=misfit)
        k = np.array([5.0, -2.0, 7.0])
        khat = k/np.sqrt(k.dot(k))
        zeroth = strain.zeroth_order_green_function(khat)
        self.assertTrue(np.allclose(zeroth, self.isotropic_green_function(khat)))

    def eshelby_strain_energy_sphere(self, misfit):
        return 2*(1+self.poisson)*self.G*misfit**2/(1-self.poisson)

    def eshelby_strain_energy_plate(self, misfit):
        return 2*(1+self.poisson)*self.G*misfit**2/(1-self.poisson)

    def eshelby_strain_energy_needle(self, misfit):
        return 2*(1+self.poisson)*self.G*misfit**2/(1-self.poisson)

    def test_green_function_cpp(self):
        if not available:
            self.skipTest(available)
        misfit = np.eye(3)*0.05
        ft = np.zeros((8, 8, 8))
        elastic = to_full_rank4(self.get_isotropic_tensor())
        pykhach = PyKhachaturyan(ft, elastic, misfit)
        k = np.array([-1.0, 3.0, 2.5])
        k /= np.sqrt(k.dot(k))
        gf = pykhach.green_function(k)
        self.assertTrue(np.allclose(gf, self.isotropic_green_function(k)))

    def test_frequency(self):
        if not available:
            self.skipTest(reason)
        misfit = np.eye(3)*0.05
        ft = np.zeros((8, 8, 8))
        elastic = to_full_rank4(self.get_isotropic_tensor())
        pykhach = PyKhachaturyan(ft, elastic, misfit)

        freq = np.fft.fftfreq(ft.shape[0])
        for i in range(ft.shape[0]):
            indx = np.array([i, 0, 0])
            self.assertAlmostEqual(freq[i], pykhach.wave_vector(indx)[0])

    def test_sphere(self):
        if not available:
            self.skipTest(reason)
        eps = 0.05
        misfit = np.eye(3)*eps
        strain = Khachaturyan(elastic_tensor=self.get_isotropic_tensor(),
                              misfit_strain=misfit)
        sph = self.get_sphere_voxels(256)
        E = strain.strain_energy_voxels(sph)
        E_eshelby = self.eshelby_strain_energy_sphere(eps)
        self.assertAlmostEqual(E, E_eshelby, places=3)

    def test_sphere_pure_python(self):
        if not available:
            self.skipTest(reason)
        eps = 0.05
        misfit = np.eye(3)*eps
        strain = Khachaturyan(elastic_tensor=self.get_isotropic_tensor(),
                              misfit_strain=misfit)
        sph = self.get_sphere_voxels(32)
        E = strain.strain_energy_voxels(sph)
        E_eshelby = self.eshelby_strain_energy_sphere(eps)
        self.assertAlmostEqual(E, E_eshelby, places=3)

    def test_plate_voxels(self):
        if not available:
            self.skipTest(reason)
        
        eps = 0.05
        misfit = np.eye(3)*eps
        strain = Khachaturyan(elastic_tensor=self.get_isotropic_tensor(),
                              misfit_strain=misfit)
        plate = self.get_plate_voxels(256)
        E = strain.strain_energy_voxels(plate)
        E_eshelby = self.eshelby_strain_energy_plate(eps)
        self.assertAlmostEqual(E, E_eshelby, places=3)

    def test_needle_voxels(self):
        if not available:
            self.skipTest(reason)
        
        eps = 0.05
        misfit = np.eye(3)*eps
        strain = Khachaturyan(elastic_tensor=self.get_isotropic_tensor(),
                              misfit_strain=misfit)
        needle = self.get_needle_voxels(256)
        E = strain.strain_energy_voxels(needle)
        E_eshelby = self.eshelby_strain_energy_needle(eps)
        self.assertAlmostEqual(E, E_eshelby, places=3)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)

