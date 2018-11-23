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
        from ase.visualize import view
        view(atoms)

        gaussian_ct = GaussianClusterTracker(atoms=atoms, threshold=0.01, cluster_elements=["Mg"],
                                             num_clusters=2)
        gaussian_ct.find_clusters()
        gaussian_ct.show_clusters()
        

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
        
        
