import unittest
try:
    import dataset
    import numpy as np
    from cemc.tools.phasediagram import BinaryPhaseDiagram
    available = True
    reason = ""
except ImportError as exc:
    reason = str(exc)
    available = False


class TestBinaryPhaseDiag(unittest.TestCase):
    db_name = "sqlite:///binary_phase_diag.db"

    def setUp(self):
        self.prepare_db()

    def prepare_db(self):
        mu = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        phases = ["Al", "Mg"]
        db = dataset.connect(self.db_name)
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
        mu = 0.1
        t = [200, 300, 400, 500, 600, 700, 800]
        for T in t:
            data = {
                    "mu": m,
                    "conc": concs[ph],
                    "temperature": T,
                    "energy": E0 + mu,
                    "phase": "Al"
                }
            tbl.insert(data)

        # Insert random phase
        kB = 8.6E-5
        for T in t:
            data = {
                    "mu": m,
                    "conc": concs[ph],
                    "temperature": T,
                    "energy": E0 - np.log(2)*T*kB,
                    "phase": "random"
                }
            tbl.insert(data)

    def test_fixed_temperature(self):
        if not available:
            self.skipTest(reason)

        phase_diag = BinaryPhaseDiagram(
            db_name=self.db_name, table="simulations",
            energy="energy", concentration="conc",
            num_elem=2, natoms=1, chem_pot="mu",
            recalculate_postproc=True)

        self.assertEqual(sorted(["Al", "Mg", "random"]),
                         sorted(phase_diag.all_phases))

        inter = phase_diag.fixed_temperature_line(
            temperature=400, phases=["Al", "Mg"], polyorder=1)
        self.assertFalse(inter is None)
        self.assertAlmostEqual(inter, 0.0)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
