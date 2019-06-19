import unittest
try:
    from ase.clease.settings_bulk import CEBulk
    from ase.clease import CorrFunction
    from ase.calculators.clease import Clease
    from ase.clease import Concentration
    from cemc.mfa.mean_field_approx import MeanFieldApprox
    from cemc.mfa import CanonicalMeanField
    from ase.units import kB
    from cemc import CE
    has_ase_with_ce = True
except Exception as exc:
    print ( str(exc) )
    has_ase_with_ce = False

import numpy as np

class TestMFA( unittest.TestCase ):
    def test_no_throw(self):
        if ( not has_ase_with_ce ):
            self.skipTest( "The ASE version does not include the CE module!" )
            return

        no_throw = True
        msg = ""
        try:
            db_name = "test_db.db"
            conc = Concentration(basis_elements=[["Al","Mg"]])
            ceBulk = CEBulk(
                crystalstructure="fcc",
                a=4.05, size=[3, 3, 3],
                concentration=conc, db_name=db_name, max_cluster_size=3,
                max_cluster_dia=4.5)
            ceBulk.reconfigure_settings()
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)
            ecis = {key:0.001 for key in cf.keys()}
            atoms = ceBulk.atoms.copy()
            calc = Clease( ceBulk, cluster_name_eci=ecis ) # Bug in the update
            atoms.set_calculator(calc)
            #ceBulk.atoms.set_calculator( calc )
            mf = MeanFieldApprox(atoms, ceBulk)
            chem_pot = {"c1_0":-1.05}
            betas = np.linspace( 1.0/(kB*100), 1.0/(kB*800), 50 )
            G = mf.free_energy( betas, chem_pot=chem_pot)
            G = mf.free_energy( betas ) # Try when chem_pot is not given
            U = mf.internal_energy( betas, chem_pot=chem_pot )
            U = mf.internal_energy( betas )
            Cv = mf.heat_capacity( betas, chem_pot=chem_pot )
            Cv = mf.heat_capacity( betas )

            atoms = ceBulk.atoms.copy()
            atoms[0].symbol = "Mg"
            atoms[1].symbol = "Mg"
            calc = CE(atoms, ceBulk, eci=ecis)
            #ceBulk.atoms.set_calculator(calc)
            # Test the Canonical MFA
            T = [500, 400, 300, 200, 100]
            canonical_mfa = CanonicalMeanField(atoms=atoms, T=T)
            res = canonical_mfa.calculate()
        except Exception as exc:
            no_throw = False
            msg = str(exc)
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
