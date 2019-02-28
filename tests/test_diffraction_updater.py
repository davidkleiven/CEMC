import unittest
import numpy as np

try:
    from cemc.mcmc.diffraction_observer import DiffractionUpdater
    from ase.build import bulk
    from ase.geometry import get_layers
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestDiffractionUpdater(unittest.TestCase):
    def test_layered(self):
        if not available:
            self.skipTest(reason)

        atoms = bulk("Al", crystalstructure="sc", a=5.0)
        atoms *= (10, 10, 10)
        layers, dist = get_layers(atoms, (1, 0, 0))
        for atom in atoms:
            if layers[atom.index] % 2 == 0:
                atom.symbol = "Al"
            else:
                atom.symbol = "Mg"

        k = 2.0*np.pi/10.0
        updater = DiffractionUpdater(atoms=atoms, k_vector=[k, 0, 0],
                                     active_symbols=["Mg"],
                                     all_symbols=["Al", "Mg"])
        self.assertAlmostEqual(np.abs(updater.value), 0.5)

        updater = DiffractionUpdater(atoms=atoms, k_vector=[0, k, 0],
                                     active_symbols=["Mg"],
                                     all_symbols=["Al", "Mg"])
        self.assertAlmostEqual(np.abs(updater.value), 0.0)

    def test_update_method(self):
        if not available:
            self.skipTest(reason)

        atoms = bulk("Al", crystalstructure="sc", a=5.0)
        atoms *= (10, 10, 10)
        layers, dist = get_layers(atoms, (1, 0, 0))
        for atom in atoms:
            if layers[atom.index] % 2 == 0:
                atom.symbol = "Al"
            else:
                atom.symbol = "Mg"

        k = 2.0*np.pi/10.0
        updater = DiffractionUpdater(atoms=atoms, k_vector=[k, 0, 0],
                                     active_symbols=["Mg"],
                                     all_symbols=["Al", "Mg"])
        
        # Remove all Al atoms
        for i in range(len(atoms)):
            if atoms[i].symbol == "Al":
                system_change = [(i, "Al", "Mg")]
                updater.update(system_change)

        # Reflection should be zero
        self.assertAlmostEqual(np.abs(updater.value), 0.0)

        # Insert Al atoms again
        for i in range(len(atoms)):
            if layers[i] % 2 == 0:
                system_change = [(i, "Mg", "Al")]
                updater.update(system_change)
        
        # Now value should be back to 1/2
        self.assertAlmostEqual(np.abs(updater.value), 0.5)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
