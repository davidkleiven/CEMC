import unittest
from ase.ce.settings import BulkCrystal
from ase.ce.corrFunc import CorrFunction
from wanglandau.ce_calculator import CE
import numpy as np

class TestCE( unittest.TestCase ):
    def test_update( self ):
        db_name = "test_db.db"
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        eci = {}
        ceBulk = BulkCrystal( "fcc", 4.05, None, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
        calc = CE( ceBulk, eci )
        calc.eci = {key:1.0 for key in calc.cf}
        eci = calc.eci
        n_tests = 10
        for i in range(n_tests):
            calc.update_cf( (i+1, "Al", "Mg") )
            updated_cf = calc.get_cf()
            calc = CE( ceBulk, eci ) # Now the calculator is initialized with the new atoms object
            for key,value in updated_cf.iteritems():
                self.assertAlmostEqual( value, calc.cf[key] )

    def test_random_swaps( self ):
        db_name = "test_db.db"
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        ceBulk = BulkCrystal( "fcc", 4.05, None, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
        corr_func = CorrFunction(ceBulk)
        cf = corr_func.get_cf( ceBulk.atoms )
        eci = {name:1.0 for name in cf.keys()}
        calc = CE( ceBulk, eci )
        n_tests = 100

        for i in range(n_tests):
            print ("%d of %d random tests"%(i+1,n_tests))
            indx = np.random.randint(low=0,high=len(ceBulk.atoms))
            old_symb = ceBulk.atoms[indx].symbol
            if ( old_symb == "Al" ):
                new_symb = "Mg"
            else:
                new_symb = "Al"
            calc.calculate( ceBulk.atoms, ["energy"], [(indx, old_symb, new_symb)] )
            updated_cf = calc.get_cf()
            brute_force = corr_func.get_cf_by_cluster_names( ceBulk.atoms, eci.keys() )
            for key,value in updated_cf.iteritems():
                self.assertAlmostEqual( value, brute_force[key] )


    def test_double_swaps( self ):
        db_name = "test_db.db"
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        ceBulk = BulkCrystal( "fcc", 4.05, None, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
        corr_func = CorrFunction(ceBulk)
        cf = corr_func.get_cf(ceBulk.atoms)
        eci = {name:1.0 for name in cf.keys()}
        calc = CE( ceBulk, eci )
        ceBulk.atoms.set_calculator(calc)
        n_tests = 100

        # Insert 25 Mg atoms
        for i in range(25):
            calc.calculate( ceBulk.atoms, ["energy"], [(i,"Al","Mg")] )

        # Swap Al and Mg atoms
        for i in range(n_tests):
            indx1 = np.random.randint(low=0,high=len(ceBulk.atoms))
            symb1 = ceBulk.atoms[indx1].symbol
            indx2 = indx1
            symb2 = symb1
            while( symb2 == symb1 ):
                indx2 = np.random.randint(low=0,high=len(ceBulk.atoms))
                symb2 = ceBulk.atoms[indx2].symbol
            print (indx1,symb1,indx2,symb2)
            calc.calculate( ceBulk.atoms, ["energy"], [(indx1,symb1,symb2),(indx2,symb2,symb1)] )
            updated_cf = calc.get_cf()
            brute_force = corr_func.get_cf_by_cluster_names( ceBulk.atoms, eci.keys() )
            for key,value in brute_force.iteritems():
                self.assertAlmostEqual( value, updated_cf[key] )

if __name__ == "__main__":
    unittest.main()
