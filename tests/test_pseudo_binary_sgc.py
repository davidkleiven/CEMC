import unittest
import os

avail_msg = ""
try:
    from cemc.mcmc import PseudoBinarySGC
    from cemc import CE
    from helper_functions import get_ternary_BC, get_example_ecis
    available = True
except ImportError as exc:
    avail_msg = str(exc)
    print(avail_msg)
    available = False

class TestPseudoBinary(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest(avail_msg)

        msg = ""
        try:
            os.remove("test_db_ternary.db")
        except Exception:
            pass
        no_throw = True
        try:
            bc = get_ternary_BC()
            ecis = get_example_ecis(bc=bc)
            calc = CE(bc, eci=ecis)
            bc.atoms.set_calculator(calc)

            groups = [{"Al": 2}, {"Mg": 1, "Si": 1}]
            symbs = ["Al", "Mg", "Si"]
            T = 400
            mc = PseudoBinarySGC(bc.atoms, T, chem_pot=-0.2, symbols=symbs,
                                 groups=groups)
            mc.runMC(mode="fixed", steps=100, equil=False)
            os.remove("test_db_ternary.db")
        except Exception as exc:
            msg = "{}: {}".format(type(exc).__name__, str(exc))
            no_throw = False
        self.assertTrue(no_throw, msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
