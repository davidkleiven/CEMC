import unittest
try:
    from cemc.mcmc import GaussianClusterTracker
    from ase.build import bulk
    available = True
    skip_msg = ""
    from ase.cluster.cubic import FaceCenteredCubic
    import numpy as np
except Exception as exc:
    available = False
    skip_msg = str(exc)
    print(skip_msg)

class TestGaussianClusterTracker(unittest.TestCase):
    def bulk_with_nano_particle(self):
        atoms = bulk("Al", a=4.05)*(10, 10, 10)
        surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        layers = [4, 7, 3]
        lc = 4.05
        nano_part = FaceCenteredCubic('Mg', surfaces, layers, latticeconstant=lc)

        nano_part_pos = nano_part.get_positions()
        com = np.mean(nano_part_pos, axis=0)
        nano_part_pos -= com

        cell = atoms.get_cell()
        diag = 0.5*np.sum(cell, axis=0)
        
        atoms_pos = atoms.get_positions() - diag

        # Insert the nano particles
        for i in range(nano_part_pos.shape[0]):
            diff = atoms_pos - nano_part_pos[i, :]
            lengths = np.sum(diff**2, axis=1)
            indx = np.argmin(lengths)
            atoms[indx].symbol = nano_part[i].symbol
        return atoms

    def test_cluster_tracker(self):
        if not available:
            self.skipTest(skip_msg)
        atoms = self.bulk_with_nano_particle()
        gaussian_ct = GaussianClusterTracker(atoms=atoms, threshold=0.01, cluster_elements=["Mg"])
        gaussian_ct.find_clusters()
        self.assertEqual(gaussian_ct.num_clusters, 1)

    def test_more_clusters(self):
        if not available:
            self.skipTest(skip_msg)
        atoms = bulk("Al", a=4.05)*(10, 10, 10)
        for i in range(30):
            atoms[i].symbol = "Mg"
            atoms[-i].symbol = "Mg"
        centroids = [[0.0, 0.0, 0.0], [18, 18, 36]]
        gaussian_ct = GaussianClusterTracker(atoms=atoms, threshold=0.0001, cluster_elements=["Mg"],
                                             num_clusters=2, init_centroids=centroids)
        gaussian_ct.find_clusters()
        cluster1 = np.array(range(0, 30))
        cluster2 = np.array(range(971, 1000))
        self.assertTrue(np.all(cluster1 == gaussian_ct.get_cluster(0)))

        self.assertTrue(np.all(cluster2 == gaussian_ct.get_cluster(1)))

        try:
            gaussian_ct.show_gaussians(scale=2.0, show=False)
        except ImportError:
            pass
        except Exception as exc:
            self.fail(str(exc))

    def test_updates(self):
        if not available:
            self.skipTest(skip_msg)

        atoms = bulk("Al", a=4.05)*(10, 10, 10)
        for i in range(30):
            atoms[i].symbol = "Mg"
        gaussian_ct = GaussianClusterTracker(atoms=atoms, threshold=0.0001, cluster_elements=["Mg"],
                                            num_clusters=1)  
        gaussian_ct.find_clusters()

        N = len(atoms)
        sigma_orig = gaussian_ct.gaussians[0].sigma
        mu_orig = gaussian_ct.gaussians[0].mu
        pos = atoms[:30].get_positions()
        self.assertTrue(np.allclose(np.mean(pos, axis=0), mu_orig))

        # Test single change. Extend cluster
        syst_change = [(60, "Al", "Mg")]
        gaussian_ct.update_clusters(syst_change)
        atoms[60].symbol = "Mg"
        sigma = gaussian_ct.gaussians[0].sigma.copy()
        mu = gaussian_ct.gaussians[0].mu.copy()
        gaussian_ct.find_clusters()
        sigma_exc = gaussian_ct.gaussians[0].sigma
        mu_exc = gaussian_ct.gaussians[0].mu
        self.assertTrue(np.allclose(mu_exc, mu))
        self.assertTrue(np.allclose(sigma_exc, sigma))

        # Test single change. Remove element
        syst_change = [(0, "Mg", "Al")]
        gaussian_ct.update_clusters(syst_change)
        atoms[0].symbol = "Al"
        sigma = gaussian_ct.gaussians[0].sigma.copy()
        mu = gaussian_ct.gaussians[0].mu.copy()
        gaussian_ct.find_clusters()
        sigma_exc = gaussian_ct.gaussians[0].sigma
        mu_exc = gaussian_ct.gaussians[0].mu
        self.assertTrue(np.allclose(mu_exc, mu))
        self.assertTrue(np.allclose(sigma_exc, sigma))


        # Perform 100 updates and enure that the gaussian is good
        for _ in range(100):
            indx1 = np.random.randint(low=0, high=N)
            symb1 =atoms[indx1].symbol
            symb2 = symb1
            while symb2 == symb1:
                indx2 = np.random.randint(low=0, high=N)
                symb2 = atoms[indx2].symbol
            syst_change = [(indx1, symb1, symb2), (indx2, symb2, symb1)]
            gaussian_ct.update_clusters(syst_change)
            atoms[indx1].symbol = symb2
            atoms[indx2].symbol = symb1

        sigma = gaussian_ct.gaussians[0].sigma.copy()
        mu = gaussian_ct.gaussians[0].mu.copy()

        gaussian_ct.find_clusters()
        sigma_exc = gaussian_ct.gaussians[0].sigma
        mu_exc = gaussian_ct.gaussians[0].mu

        self.assertTrue(np.allclose(sigma, sigma_exc))
        self.assertTrue(np.allclose(mu, mu_exc))

        

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
        
        
