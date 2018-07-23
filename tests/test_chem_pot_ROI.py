import unittest
skip_msg = ""
try:
    from helper_functions import get_small_BC_with_ce_calc
    from cemc.tools import ChemicalPotentialROI
    available = True
except Exception as exc:
    available = False
    print (str(exc))
    skip_msg = "Not available because "+str(exc)


class TestChemPotROI(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest(skip_msg)
            return
        no_throw = True
        msg = ""

        try:
            bc = get_small_BC_with_ce_calc()
            roi = ChemicalPotentialROI(bc.atoms, symbols=["Al","Mg"])
            chem_pots = roi.chemical_potential_roi()
            sampling, names = roi.suggest_mu(mu_roi=chem_pots,N=10)
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
