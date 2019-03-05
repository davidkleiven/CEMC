import unittest
from ase.build import bulk
try:
    from cemc.mcmc import ConcentrationObserver
    available = True
    reason = ""
except ImportError as exc:
    reason = str(exc)


class TestConcentrationObserver(unittest.TestCase):
    def test_peak(self):
        N = 10
        atoms = bulk("Al")*(N, N, N)
        num_atoms = len(atoms)

        conc_obs = ConcentrationObserver(symbols=["Al", "Mg", "Si"],
                                         atoms=atoms)

        # Insert one Mg atom
        val = conc_obs([(0, "Al", "Mg")], peak=True)
        expected = {"Al": 999.0/1000.0, "Mg": 1.0/1000.0, "Si": 0.0}
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])

        # Current value should remain unchanged
        expected = {"Al": 1.0, "Mg": 0.0, "Si": 0.0}
        val = conc_obs.get_current_value()
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])

        # Try a more complicated
        changes = [(0, "Al", "Mg"), (1, "Al", "Si"), (2, "Al", "Mg"),
                   (3, "Al", "Si")]
        val = conc_obs(changes, peak=True)
        expected = {"Al": 996.0/1000.0, "Mg": 2.0/1000.0, "Si": 2.0/1000.0}
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])

        # Current value should remain unchanged
        expected = {"Al": 1.0, "Mg": 0.0, "Si": 0.0}
        val = conc_obs.get_current_value()
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])

    def test_update(self):
        N = 10
        atoms = bulk("Al")*(N, N, N)
        num_atoms = len(atoms)

        conc_obs = ConcentrationObserver(symbols=["Al", "Mg", "Si"],
                                         atoms=atoms)

        # Insert one Mg atom
        val = conc_obs([(0, "Al", "Mg")], peak=False)
        expected = {"Al": 999.0/1000.0, "Mg": 1.0/1000.0, "Si": 0.0}
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])

        # Current value should now also change
        val = conc_obs.get_current_value()
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])
        conc_obs.reset()

        # Try a more complicated
        changes = [(0, "Al", "Mg"), (1, "Al", "Si"), (2, "Al", "Mg"),
                   (3, "Al", "Si")]
        val = conc_obs(changes, peak=False)
        expected = {"Al": 996.0/1000.0, "Mg": 2.0/1000.0, "Si": 2.0/1000.0}
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])

        # Current value should now also change
        val = conc_obs.get_current_value()
        for k, v in val.items():
            self.assertAlmostEqual(v, expected[k])


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)