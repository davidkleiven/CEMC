import unittest
import os
na_msg = ""
try:
    from cemc.mcmc import DamageSpreadingMC
    from ase.ce import BulkCrystal
    from helper_functions import get_max_cluster_dia_name
    from cemc import CE
    available = True
except ImportError as exc:
    print(str(exc))
    na_msg = str(exc)
    available = False

ecis = {
    "c1_0": -0.1,
    "c1_1": 0.1,
}

db_name = "dm_mc.db"


class TestDMMC(unittest.TestCase):
    def init_bulk_crystal(self):
        conc_args = {
            "conc_ratio_min_1": [[2, 1, 1]],
            "conc_ratio_max_1": [[0, 2, 2]],
        }
        max_dia_name = get_max_cluster_dia_name()
        size_arg = {max_dia_name: 4.05}
        ceBulk = BulkCrystal(crystalstructure="fcc", a=4.05, size=[3, 3, 3],
                             basis_elements=[["Al", "Mg", "Si"]],
                             conc_args=conc_args, db_name=db_name,
                             max_cluster_size=3, **size_arg)
        ceBulk.reconfigure_settings()
        calc = CE(ceBulk, ecis)
        ceBulk.atoms.set_calculator(calc)
        return ceBulk

    def test_no_throw(self):
        """Test that the code runs without throwing errors."""
        if not available:
            self.skipTest(na_msg)

        no_throw = True
        msg = ""
        try:
            bc1 = self.init_bulk_crystal()
            bc2 = self.init_bulk_crystal()

            dm_mc = DamageSpreadingMC(bc1.atoms, bc2.atoms, 300,
                                      symbols=["Al", "Mg", "Si"])

            chem_pot = {"c1_0": 0.0, "c1_1": 1.0}
            dm_mc.runMC(chem_pot)
            dm_mc.reset()
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        os.remove(db_name)
        self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
