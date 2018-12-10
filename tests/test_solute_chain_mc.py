import unittest
import traceback
try:
    from cemc.mcmc import SoluteChainMC
    from ase.clease import CEBulk, Concentration
    from helper_functions import get_example_network_name
    from helper_functions import get_max_cluster_dia_name
    from helper_functions import get_example_ecis
    from cemc import CE
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)

DB_NAME = "sulute_chain_mc.db"
class TestSoluteChainMC(unittest.TestCase):

    def init_bulk_crystal(self):
        max_dia_name = get_max_cluster_dia_name()
        size_arg = {max_dia_name: 4.05}
        conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])
        ceBulk = CEBulk(crystalstructure="fcc", a=4.05, size=[3, 3, 3],
                             concentration=conc, db_name=DB_NAME,
                             max_cluster_size=3, **size_arg)
        ecis = get_example_ecis(ceBulk)
        ceBulk.reconfigure_settings()
        atoms = ceBulk.atoms.copy()
        calc = CE(atoms, ceBulk, ecis)
        return ceBulk, atoms

    def test_no_throw(self):
        """Make sure SoluteChainMC runs without errors."""
        if not available:
            self.skipTest(reason)

        no_throw = True
        msg = ""
        try:
            ceBulk, atoms = self.init_bulk_crystal()

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

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)