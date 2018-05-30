import unittest
import copy
import numpy as np
try:
    from ase.ce.settings_bulk import BulkCrystal
    from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker, PhaseChangedOnFirstIterationError
    from cemc.mcmc import linear_vib_correction as lvc
    from helper_functions import get_example_cf, get_example_ecis
    has_ase_with_ce = True
except:
    has_ase_with_ce = False

ecivib = {
    "c1_0":0.42
}

db_name = "test_phase_boundary.db"
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
                "eci":get_example_ecis(bc=b1),
                "cf":get_example_cf(bc=b1)
            }

            gs2 = {
                "bc":b2,
                "eci":get_example_ecis(bc=b2),
                "cf":get_example_cf(bc=b2)
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

    def test_adaptive_euler(self):
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
                "eci":get_example_ecis(bc=b1),
                "cf":get_example_cf(bc=b1)
            }

            gs2 = {
                "bc":b2,
                "eci":get_example_ecis(bc=b2),
                "cf":get_example_cf(bc=b2)
            }

            boundary = PhaseBoundaryTracker( gs1, gs2 )
            T = [10,20]
            mc_args = {
                "steps":10,
                "mode":"fixed",
                "equil":False
            }
            res = boundary.separation_line_adaptive_euler( T0=100,min_step=99,stepsize=100, mc_args=mc_args )
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue( no_throw, msg=msg )

    def test_with_linvib( self ):
        if ( not has_ase_with_ce ):
            self.skipTest( "ASE version does not have CE" )
        try:
            b1, b2 = self.init_bulk_crystal()
            gs1 = {
                "bc":b1,
                "eci":get_example_ecis(bc=b1),
                "cf":get_example_cf(bc=b1),
                "linvib":lvc.LinearVibCorrection( eci_vib )
            }

            gs2 = {
                "bc":b2,
                "eci":get_example_ecis(bc=b2),
                "cf":get_example_cf(bc=b2),
                "linvib":lvc.LinearVibCorrection( eci_vib )
            }

            boundary = PhaseBoundaryTracker( gs1, gs2 )
            T = [10,20]
            res = boundary.separation_line( np.array(T) )
        except PhaseChangedOnFirstIterationError as exc:
            pass
        except Exception as exc:
            no_throw = False
            msg = str(exc)

if __name__ == "__main__":
    unittest.main()
