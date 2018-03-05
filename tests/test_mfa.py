import unittest
try:
    from ase.ce.settings import BulkCrystal
    from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
    from cemc.mfa.mean_field_approx import MeanFieldApprox
    from ase.units import kB
    has_ase_with_ce = True
except Exception as exc:
    print ( str(exc) )
    has_ase_with_ce = False

import numpy as np

# Some ECIs computed for the Al-Mg system
ecis = {
    "c3_1225_4_1": -0.0017448109612305434,
    "c2_1000_1_1": -0.02253231472540913,
    "c4_1225_8_1": 0.0015986520863819958,
    "c2_707_1_1": 0.0020761708499214765,
    "c4_707_1_1": -1.5475822532285122e-05,
    "c4_1225_3_1": 0.0013284466570874605,
    "c1_1": -1.068187483782512,
    "c3_1225_2_1": -0.0015608053756936988,
    "c3_1225_1_1": -0.0010685006728372629,
    "c0": -2.6460513669066836,
    "c4_1225_9_1": -7.3952137244461468e-05
}
class TestMFA( unittest.TestCase ):
    def test_no_throw(self):
        if ( not has_ase_with_ce ):
            self.skipTest( "The ASE version does not include the CE module!" )
            return

        no_throw = True
        msg = ""
        try:
            db_name = "test_db.db"
            conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
            ceBulk = BulkCrystal( "fcc", 4.05, None, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
            calc = ClusterExpansion( ceBulk, cluster_name_eci=ecis )
            ceBulk.atoms.set_calculator( calc )
            mf = MeanFieldApprox( ceBulk )
            chem_pot = {"c1_1":-1.05}
            betas = np.linspace( 1.0/(kB*100), 1.0/(kB*800), 50 )
            G = mf.free_energy( betas, chem_pot=chem_pot)
            G = mf.free_energy( betas ) # Try when chem_pot is not given
            U = mf.internal_energy( betas, chem_pot=chem_pot )
            U = mf.internal_energy( betas )
            Cv = mf.heat_capacity( betas, chem_pot=chem_pot )
            Cv = mf.heat_capacity( betas )
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    unittest.main()
