import unittest
import os
try:
    from ase.clease import CEBulk, Concentration
    from cemc.mcmc import ParallelTempering
    from cemc.mcmc import Montecarlo
    from helper_functions import get_example_ecis
    from cemc import CE
    available = True
    import_msg = ""
except ImportError as exc:
    import_msg = str(exc)
    available = False

db_name = "parallel_tempering.db"


class TestParallelTempering(unittest.TestCase):
    def init_bulk_crystal(self):
        conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])
        ceBulk = CEBulk(crystalstructure="fcc", a=4.05, size=[4, 4, 4],
                             concentration=conc, db_name=db_name,
                             max_cluster_size=2, max_cluster_dia=4.0)
        ceBulk.reconfigure_settings()
        ecis = get_example_ecis(bc=ceBulk)
        atoms = ceBulk.atoms.copy()
        calc = CE(atoms, ceBulk, ecis)
        return ceBulk, atoms

    def test_no_throw(self):
        if not available:
            self.skipTest(import_msg)

        msg = ""
        no_throw = True
        try:
            ceBulk, atoms = self.init_bulk_crystal()
            mc = Montecarlo(atoms, 100.0)
            mc.insert_symbol_random_places("Mg", num=5, swap_symbs=["Al"])
            mc.insert_symbol_random_places("Si", num=5, swap_symbs=["Al"])
            par_temp = ParallelTempering(mc_obj=mc, Tmax=100.0, Tmin=0.001)
            mc_args = {"steps": 100, "equil": False}
            par_temp.run(mc_args=mc_args, num_exchange_cycles=3)
            os.remove("temp_scheme.csv")
        except Exception as exc:
            msg = "{}: {}".format(type(exc).__name__, str(exc))
            no_throw = False
        self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
