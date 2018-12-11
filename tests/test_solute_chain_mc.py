import unittest
import traceback
import os
try:
    from cemc.mcmc import SoluteChainMC
    from ase.clease import CEBulk, Concentration
    from helper_functions import get_example_network_name
    from helper_functions import get_max_cluster_dia_name
    from helper_functions import get_example_ecis
    from cemc import get_atoms_with_ce_calc
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)

DB_NAME = "sulute_chain_mc.db"
DB_NAME_CALC = "solute_chain_mc_calc.db"
class TestSoluteChainMC(unittest.TestCase):

    def init_bulk_crystal(self):
        max_dia_name = get_max_cluster_dia_name()
        size_arg = {max_dia_name: 4.05}
        conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])
        args = dict(crystalstructure="fcc", a=4.05, size=[3, 3, 3],
                             concentration=conc, db_name=DB_NAME,
                             max_cluster_size=3, **size_arg)
        ceBulk = CEBulk(**args)
        ecis = get_example_ecis(ceBulk)
        
        atoms = get_atoms_with_ce_calc(ceBulk, args, eci=ecis, size=[10, 10, 10],
                                       db_name=DB_NAME_CALC)
        return atoms

    def test_no_throw(self):
        """Make sure SoluteChainMC runs without errors."""
        if not available:
            self.skipTest(reason)

        no_throw = True
        msg = ""
        try:
            atoms = self.init_bulk_crystal()
            ceBulk = atoms.get_calculator().BC
            name = get_example_network_name(ceBulk)
            T = 600.0
            mc = SoluteChainMC(atoms, T, cluster_elements=["Mg", "Si"], cluster_names=[name])
            mc.build_chain({"Mg": 4, "Si": 4})
            mc.runMC(steps=10000, equil=False)
        except Exception as exc:
            msg = type(exc).__name__ + str(exc)
            msg += "\n"+traceback.format_exc()
            no_throw = False
        self.assertTrue(no_throw, msg)
        os.remove(DB_NAME)
        os.remove(DB_NAME_CALC)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)