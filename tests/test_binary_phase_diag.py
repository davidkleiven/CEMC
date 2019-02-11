import unittest
import os
try:
    import dataset
    import numpy as np
    from cemc.tools.phasediagram import BinaryPhaseDiagram
    available = True
    reason = ""
except ImportError as exc:
    reason = str(exc)
    available = False

kB = 8.6E-5  # Boltzmann constant eV/K


class TestBinaryPhaseDiag(unittest.TestCase):
    db_name = "binary_phase_diag.db"

    def setUp(self):
        self.prepare_db()

    @property
    def full_db_name(self):
        return "sqlite:///"+self.db_name

    @property
    def phase_diag_instance(self):
        return BinaryPhaseDiagram(
            db_name=self.full_db_name, table="simulations",
            energy="energy", concentration="conc",
            num_elem=2, natoms=1, chem_pot="mu",
            recalculate_postproc=True, ht_phases=["random"])

    def prepare_db(self):
        mu = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        phases = ["Al", "Mg"]
        db = dataset.connect(self.full_db_name)
        tbl = db["simulations"]
        E0 = -3.74

        concs = {
            "Al": 0.05,
            "Mg": 0.95
        }

        slopes = {
            "Al": 0.1,
            "Mg": -0.2
        }
        for ph in phases:
            for m in mu:
                data = {
                    "mu": m,
                    "conc": concs[ph],
                    "temperature": 400,
                    "energy": E0 + slopes[ph]*m,
                    "phase": ph
                }
                tbl.insert(data)

        # Insert data for varying temperature
        mu = 0.05
        t = [200, 300, 400, 500]
        for T in t:
            data = {
                    "mu": mu,
                    "conc": concs[ph],
                    "temperature": T,
                    "energy": E0 + mu,
                    "phase": "Al"
                }
            tbl.insert(data)

        # Insert random phase
        t = [900, 1000, 1100, 1200]
        for T in t:
            data = {
                    "mu": mu,
                    "conc": concs[ph],
                    "temperature": T,
                    "energy": E0 + 1.2*mu,
                    "phase": "random"
                }
            tbl.insert(data)

    def test_fixed_temperature(self):
        if not available:
            self.skipTest(reason)

        phase_diag = self.phase_diag_instance

        self.assertEqual(sorted(["Al", "Mg", "random"]),
                         sorted(phase_diag.all_phases))

        inter = phase_diag.phase_intersection(
            temperature=400, phases=["Al", "Mg"], polyorder=1)
        self.assertFalse(inter is None)
        self.assertAlmostEqual(inter, 0.0)

    def test_fixed_mu(self):
        if not available:
            self.skipTest(reason)

        phase_diag = self.phase_diag_instance
        inter = phase_diag.phase_intersection(
            mu=0.05, phases=["Al", "random"], polyorder=1)
        self.assertFalse(inter is None)
        self.assertAlmostEqual(inter, 0.2*0.05/(kB*np.log(2)), places=0)

    def test_phase_boundaries(self):
        if not available:
            self.skipTest(reason)

        phase_diag = self.phase_diag_instance
        mu, T = phase_diag.phase_boundary(phases=["Al", "Mg"],
                                          variable="chem_pot", polyorder=1)
        self.assertEqual(T[0], 400)
        self.assertAlmostEqual(mu[0], 0.0)

        mu, T = phase_diag.phase_boundary(phases=["Al", "random"],
                                          variable="temperature",
                                          polyorder=1)
        self.assertAlmostEqual(mu[0], 0.05)
        self.assertAlmostEqual(T[0], 0.2*0.05/(kB*np.log(2)), places=0)

    def tearDown(self):
        os.remove(self.db_name)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
