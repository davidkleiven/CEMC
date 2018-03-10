import unittest
try:
    from ase.ce.settings_bulk import BulkCrystal
    from ase.ce.corrFunc import CorrFunction
    from cemc.wanglandau.ce_calculator import CE
    has_ase_with_ce = True
except Exception as exc:
    print (str(exc))
    has_ase_with_ce = False
import numpy as np
import os

class TestCE( unittest.TestCase ):
    lattices = ["fcc","bcc","sc"]
    def get_calc(self, lat):
        db_name = "test_db_{}.db".format(lat)

        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        eci = {}
        a = 4.05
        ceBulk = BulkCrystal( crystalstructure=lat, a=a, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, \
        db_name=db_name, max_cluster_size=4)
        ceBulk._get_cluster_information()
        calc = CE( ceBulk, eci )
        calc.eci = {key:1.0 for key in calc.cf}
        eci = calc.eci
        return calc,ceBulk,eci

    def test_update( self ):
        if ( has_ase_with_ce ):
            for lat in self.lattices:
                msg = "Failed for lattice {}".format(lat)
                calc,ceBulk,eci = self.get_calc(lat)
                cf = CorrFunction(ceBulk)
                n_tests = 10
                for i in range(n_tests):
                    print (i)
                    old_symb = ceBulk.atoms[i].symbol
                    if ( old_symb == "Al" ):
                        new_symb = "Mg"
                    else:
                        new_symb = "Al"
                    calc.update_cf( (i, old_symb, new_symb) )
                    updated_cf = calc.get_cf()
                    brute_force = cf.get_cf( ceBulk.atoms )
                    for key,value in updated_cf.iteritems():
                        self.assertAlmostEqual( value, brute_force[key], msg=msg )

    def test_random_swaps( self ):
        if ( has_ase_with_ce ):

            for lat in self.lattices:
                msg = "Failed for lattice {}".format(lat)
                calc, ceBulk,eci = self.get_calc(lat)
                n_tests = 10
                corr_func = CorrFunction(ceBulk)
                cf = corr_func.get_cf( ceBulk.atoms )
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
                        self.assertAlmostEqual( value, brute_force[key], msg=msg )


    def test_double_swaps( self ):
        if ( has_ase_with_ce ):

            for lat in self.lattices:
                calc,ceBulk,eci = self.get_calc(lat)
                corr_func = CorrFunction(ceBulk)
                cf = corr_func.get_cf(ceBulk.atoms)
                n_tests = 10

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

    def test_double_swaps_ternary( self ):
        if ( has_ase_with_ce and False ): # Disable this test
            db_name = "test_db_ternary.db"
            conc_args = {
                "conc_ratio_min_1":[[4,0,0]],
                "conc_ratio_max_1":[[0,4,0]],
                "conc_ratio_min_1":[[2,2,0]],
                "conc_ratio_max_2":[[1,1,2]]
            }
            ceBulk = bcs.BulkCrystalSymmetric( "fcc", 4.05, None, [4,4,4], 1, [["Al","Mg","Si"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
            corr_func = CorrFunction( ceBulk )
            cf = corr_func.get_cf( ceBulk.atoms )
            eci = {name:ceBulk.basis_functions[0] for name in cf.keys()}
            calc = CE( ceBulk, eci )
            n_tests = 50

            # Insert 25 Mg atoms and 25 Si atoms
            n = 18
            print (ceBulk.basis_functions)
            print (ceBulk.cluster_indx[2])
            keys = eci.keys()
            keys.sort()
            print ( keys )
            for i in range(n):
                print ( "Changing element {} of {}".format(i,n) )
                calc.calculate( ceBulk.atoms, ["energy"], [(i,"Al","Mg")])
                calc.calculate( ceBulk.atoms, ["energy"], [(i+n,"Al","Si")])
                updated_cf = calc.get_cf()
                brute_force = corr_func.get_cf_by_cluster_names( ceBulk.atoms, eci.keys() )
                for key in keys:
                    print (key,brute_force[key],updated_cf[key])
                    self.assertAlmostEqual( brute_force[key], updated_cf[key])

            # Swap atoms
            for i in range(n_tests):
                print ( "Swap {} of {}".format(i,n_tests))
                indx1 = np.random.randint(low=0,high=len(ceBulk.atoms))
                symb1 = ceBulk.atoms[indx1].symbol
                indx2 = indx1
                symb2 = symb1
                while( symb2 == symb1 ):
                    indx2 = np.random.randint( low=0, high=len(ceBulk.atoms) )
                    symb2 = ceBulk.atoms[indx2].symbol
                calc.calculate( ceBulk.atoms, ["energy"], [(indx1,symb1,symb2),(indx2,symb2,symb1)])
                update_cf = calc.get_cf()
                brute_force = corr_func.get_cf_by_cluster_names( ceBulk.atoms, eci.keys() )
                for key,value in brute_force.iteritems():
                    self.assertAlmostEqual( value, update_cf[key ])
if __name__ == "__main__":
    unittest.main()
