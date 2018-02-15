import unittest
from mcmc.sgc_montecarlo import SGCMonteCarlo
from wanglandau.ce_calculator import CE
import os
from ase.ce.settings import BulkCrystal

ecis = {
    "c1_1":-0.1,
    "c1_2":0.1,
}

db_name = "test_sgc.db"
class TestSGCMC( unittest.TestCase ):
    def test_no_throw(self):
        no_throw = True
        try:
            conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
            ceBulk = BulkCrystal( "fcc", 4.05, None, [3,3,3], 1, [["Al","Mg","Si"]], conc_args, db_name, reconf_db=False)
            calc = CE( ceBulk, ecis )
            ceBulk.atoms.set_calculator(calc)
            chem_pots = {
                "c1_1":0.02,
                "c1_2":-0.03
            }
            T = 600.0
            mc = SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg","Si"] )
            mc.runMC( steps=100, chem_potential=chem_pots )
            thermo = mc.get_thermodynamic()
        except Exception as exc:
            print ( str(exc) )
            no_throw = False
        self.assertTrue( no_throw )

    def __del__( self ):
        os.remove( db_name )

if __name__ == "__main__":
    unittest.main()