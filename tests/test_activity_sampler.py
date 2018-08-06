
import unittest
try:
    from cemc.mcmc import ActivitySampler
    from cemc import CE
    from ase.ce import BulkCrystal
    from ase.ce import CorrFunction
    available = True
    reason = ""
except Exception as exc:
    reason = str(exc)
    print(reason)
    available = False


class TestActivitySampler(unittest.TestCase):
    def test_no_throw(self):
        no_throw = True
        if not available:
            self.skipTest(reason)
            return
        msg = ""
        try:
            conc_args = {
                        "conc_ratio_min_1": [[1, 0]],
                        "conc_ratio_max_1": [[0, 1]],
                    }
            kwargs = {
                "crystalstructure": "fcc",
                "a": 4.05, "size": [4, 4, 4],
                "basis_elements": [["Al", "Mg"]],
                "conc_args": conc_args,
                "db_name": "data/temporary_bcnucleationdb.db",
                "max_cluster_size": 3
            }
            ceBulk = BulkCrystal(**kwargs)
            ceBulk.reconfigure_settings()
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)
            ecis = {key: 1.0 for key in cf.keys()}
            calc = CE(ceBulk, ecis)
            ceBulk = calc.BC
            ceBulk.atoms.set_calculator(calc)

            T = 500
            c_mg = 0.4
            comp = {"Mg": c_mg, "Al": 1.0-c_mg}
            calc.set_composition(comp)
            act_sampler = ActivitySampler(ceBulk.atoms, T,
                                          moves=[("Al", "Mg")])
            act_sampler.runMC(mode="fixed", steps=1000)
            act_sampler.get_thermodynamic()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
