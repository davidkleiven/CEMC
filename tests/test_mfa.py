import unittest
try:
    from ase.ce.settings_bulk import BulkCrystal
    from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
    from cemc.mfa.mean_field_approx import MeanFieldApprox
    from cemc.mfa import CanonicalMeanField
    from ase.units import kB
    from cemc.wanglandau.ce_calculator import CE
    has_ase_with_ce = True
except Exception as exc:
    print ( str(exc) )
    has_ase_with_ce = False

import numpy as np

# Some ECIs computed for the Al-Mg system
ecis = {"c3_2000_5_000": -0.000554493287657111,
"c2_1000_1_00": 0.009635318249739103,
"c3_2000_3_000": -0.0012517824048219194,
"c3_1732_1_000": -0.0012946400900521093,
"c2_1414_1_00": -0.017537890489630819,
"c4_1000_1_0000": -1.1303654231631574e-05,
"c3_2000_4_000": -0.00065595035208737659,
"c2_1732_1_00": -0.0062866523139031511,
"c4_2000_11_0000": 0.00073748615657533178,
"c1_0": -1.0685540954294481,
"c4_1732_8_0000": 6.2192225273001889e-05,
"c3_1732_4_000": -0.00021105632231802613,
"c2_2000_1_00": -0.0058771555942559303,
"c4_2000_12_0000": 0.00026998290577185763,
"c0": -2.6460470182744342,
"c4_2000_14_0000": 0.00063004101881374334,
"c4_1414_1_0000": 0.00034847251116721441}

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
            ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], basis_elements=[["Al","Mg"]], conc_args=conc_args, db_name=db_name, max_cluster_size=4)
            ceBulk._get_cluster_information()
            calc = ClusterExpansion( ceBulk, cluster_name_eci=ecis ) # Bug in the update
            ceBulk.atoms.set_calculator( calc )
            mf = MeanFieldApprox( ceBulk )
            chem_pot = {"c1_0":-1.05}
            betas = np.linspace( 1.0/(kB*100), 1.0/(kB*800), 50 )
            G = mf.free_energy( betas, chem_pot=chem_pot)
            G = mf.free_energy( betas ) # Try when chem_pot is not given
            U = mf.internal_energy( betas, chem_pot=chem_pot )
            U = mf.internal_energy( betas )
            Cv = mf.heat_capacity( betas, chem_pot=chem_pot )
            Cv = mf.heat_capacity( betas )

            ceBulk.atoms[0].symbol = "Mg"
            ceBulk.atoms[1].symbol = "Mg"
            calc = CE( ceBulk, eci=ecis )
            ceBulk.atoms.set_calculator(calc)
            # Test the Canonical MFA
            T = [500,400,300,200,100]
            canonical_mfa = CanonicalMeanField( atoms=ceBulk.atoms, T=T )
            res = canonical_mfa.calculate()
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    unittest.main()
