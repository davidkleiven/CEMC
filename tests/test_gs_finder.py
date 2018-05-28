import unittest
try:
    from cemc.tools import GSFinder
    from ase.ce import BulkCrystal
    from ase.ce import CorrFunction
    has_CE = True
except ImportError:
    has_CE = False

class TestGSFinder( unittest.TestCase ):
    def test_gs_finder( self ):
        if ( not has_CE ):
            self.skipTest("ASE version does not have CE")
            return
        no_throw = True
        msg = ""
        try:
            db_name = "test_db_gsfinder.db"

            conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
            a = 4.05
            ceBulk = BulkCrystal( crystalstructure="fcc", a=a, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, \
            db_name=db_name, max_cluster_size=4)
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)
            eci = {key:1.0 for key in cf.keys()}
            gsfinder = GSFinder()
            comp = {"Al":0.5,"Mg":0.5}
            gsfinder.get_gs( ceBulk, eci, composition=comp )
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    unittest.main()
