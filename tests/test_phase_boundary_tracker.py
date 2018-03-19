import unittest
import copy
import numpy as np
try:
    from ase.ce.settings_bulk import BulkCrystal
    from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker, PhaseChangedOnFirstIterationError
    has_ase_with_ce = True
except:
    has_ase_with_ce = False

eci = {
    "c1_0":-0.1,
    "c2_1000_1_00":0.1,
}

cf1 = {
    "c1_0":1.0,
    "c2_1000_1_00":1.0
}
cf2 = {
    "c1_0":-1.0,
    "c2_1000_1_00":1.0
}

db_name = "test_sgc.db"
class TestPhaseBoundaryMC( unittest.TestCase ):
    def init_bulk_crystal(self):
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        ceBulk1 = BulkCrystal( crystalstructure="fcc", a=4.05, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, db_name=db_name)
        ceBulk2 = BulkCrystal( crystalstructure="fcc", a=4.05, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, db_name=db_name)
        for atom in ceBulk2.atoms:
            atom.symbol = "Mg"
        return ceBulk1, ceBulk2

    def test_no_throw( self ):
        """
        No throw
        """
        if ( not has_ase_with_ce ):
            msg = "ASE version does not have CE"
            self.skipTest( msg )
            return
        no_throw = True
        msg = ""
        try:
            b1, b2 = self.init_bulk_crystal()
            gs1 = {
                "bc":b1,
                "eci":eci,
                "cf":cf1
            }

            gs2 = {
                "bc":b2,
                "eci":eci,
                "cf":cf2
            }

            boundary = PhaseBoundaryTracker( gs1, gs2 )
            T = [10,20]
            res = boundary.separation_line( np.array(T) )
        except PhaseChangedOnFirstIterationError as exc:
            pass
        except Exception as exc:
            no_throw = False
            msg = str(exc)

        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    unittest.main()
