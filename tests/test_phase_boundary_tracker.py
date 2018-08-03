import unittest
import copy
import numpy as np
try:
    from ase.ce.settings_bulk import BulkCrystal
    from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker, PhaseChangedOnFirstIterationError
    from cemc.tools import save_phase_boundary, process_phase_boundary
    from cemc.mcmc import linear_vib_correction as lvc
    from helper_functions import get_example_cf, get_example_ecis
    from helper_functions import get_ternary_BC
    has_ase_with_ce = True
except Exception as exc:
    print(str(exc))
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
        ceBulk1.reconfigure_settings()
        ceBulk2 = BulkCrystal( crystalstructure="fcc", a=4.05, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, db_name=db_name)
        ceBulk2.reconfigure_settings()

        for atom in ceBulk2.atoms:
            atom.symbol = "Mg"
        return ceBulk1, ceBulk2

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
            ground_states = [gs1, gs2]

            boundary = PhaseBoundaryTracker( ground_states )
            T = [10,20]
            mc_args = {
                "steps":10,
                "mode":"fixed",
                "equil":False
            }
            res = boundary.separation_line_adaptive_euler( init_temp=100, min_step=99,stepsize=100, mc_args=mc_args, symbols=["Al","Mg"] )
            fname = "test_phase_boundary.h5"
            save_phase_boundary(fname, res)
            process_phase_boundary(fname)
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue( no_throw, msg=msg )

    def test_with_ternay(self):
        if ( not has_ase_with_ce ):
            msg = "ASE version does not have CE"
            self.skipTest( msg )
            return
        no_throw = True
        msg = ""
        try:
            ground_states = []
            elements = ["Al", "Mg", "Si"]
            for i in range(3):
                ce_bulk = get_ternary_BC()
                eci = get_example_ecis(bc=ce_bulk)
                for atom in ce_bulk.atoms:
                    atom.symbol = elements[i]
                gs = {
                    "bc":ce_bulk,
                    "eci":eci,
                    "cf":get_example_cf(bc=ce_bulk)
                }
                ground_states.append(gs)

            boundary = PhaseBoundaryTracker( ground_states )
            T = [10,20]
            mc_args = {
                "steps":10,
                "mode":"fixed",
                "equil":False
            }
            res = boundary.separation_line_adaptive_euler( init_temp=100,min_step=99,stepsize=100, mc_args=mc_args, symbols=["Al","Mg","Si"] )
            fname = "test_phase_boundary_ternary.h5"
            save_phase_boundary(fname, res)
            process_phase_boundary(fname)
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
