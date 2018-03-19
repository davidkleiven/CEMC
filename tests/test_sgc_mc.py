import unittest
import os
from mpi4py import MPI
from cemc.mcmc import linear_vib_correction as lvc

try:
    from ase.ce.settings_bulk import BulkCrystal
    from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
    from cemc.wanglandau.ce_calculator import CE
    has_ase_with_ce = True
except Exception as exc:
    print (str(exc))
    has_ase_with_ce = False

ecis = {
    "c1_0":-0.1,
    "c1_1":0.1,
}

db_name = "test_sgc.db"
class TestSGCMC( unittest.TestCase ):
    def init_bulk_crystal(self):
        conc_args = {
            "conc_ratio_min_1":[[2,1,1]],
            "conc_ratio_max_1":[[0,2,2]],
        }
        ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[3,3,3], basis_elements=[["Al","Mg","Si"]], conc_args=conc_args, db_name=db_name, \
        max_cluster_size=4, max_cluster_dia=4.05)
        ceBulk._get_cluster_information()
        calc = CE( ceBulk, ecis )
        ceBulk.atoms.set_calculator(calc)
        return ceBulk

    def test_no_throw(self):
        if ( not has_ase_with_ce ):
            self.skipTest( "ASE version does not have CE" )
            return
        no_throw = True
        msg = ""
        try:
            ceBulk = self.init_bulk_crystal()
            chem_pots = {
                "c1_0":0.02,
                "c1_1":-0.03
            }
            T = 600.0
            mc = SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg","Si"] )
            mc.runMC( steps=100, chem_potential=chem_pots )
            thermo = mc.get_thermodynamic()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

    def test_no_throw_mpi(self):
        if ( has_ase_with_ce ):
            no_throw = True
            msg = ""
            try:
                ceBulk = self.init_bulk_crystal()
                chem_pots = {
                    "c1_0":0.02,
                    "c1_1":-0.03
                }
                T = 600.0
                comm = MPI.COMM_WORLD
                mc = SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg","Si"], mpicomm=comm )
                mc.runMC( steps=100, chem_potential=chem_pots )
                thermo = mc.get_thermodynamic()
            except Exception as exc:
                msg = str(exc)
                no_throw = False
            self.assertTrue( no_throw, msg=msg )

    def test_no_throw_prec_mode( self ):
        if ( not has_ase_with_ce ):
            self.skipTest( "ASE version does not have CE" )
            return
        no_throw = True
        msg = ""
        try:
            ceBulk = self.init_bulk_crystal()
            chem_pots = {
                "c1_0":0.02,
                "c1_1":-0.03
            }
            T = 600.0
            mc = SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg","Si"], plot_debug=False )
            mc.runMC( chem_potential=chem_pots, mode="prec", prec_confidence=0.05, prec=10.0 )
            thermo = mc.get_thermodynamic()

            eci_vib={"c1_0":0.0}
            vib_corr = lvc.LinearVibCorrection(eci_vib)
            mc.linear_vib_correction = vib_corr
            mc.runMC( chem_potential=chem_pots, mode="prec", prec_confidence=0.05, prec=10.0 )
            thermo = mc.get_thermodynamic()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

    def __del__( self ):
        if ( os.path.isfile(db_name) ):
            os.remove( db_name )

if __name__ == "__main__":
    unittest.main()
