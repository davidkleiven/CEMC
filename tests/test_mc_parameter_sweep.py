import unittest
try:
    from ase.clease.settings_bulk import CEBulk
    from ase.clease import Concentration
    from ase.clease import CorrFunction
    from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
    from cemc import CE
    from cemc.mcmc.mc_parameter_sweep import MCParameterSweep
    has_ase_with_ce = True
except Exception as exc:
    print (str(exc))
    has_ase_with_ce = False


db_name = "test_sgc.db"
class TestMCParameterSweep( unittest.TestCase ):
    def get_cebulk(self):
        conc = Concentration(basis_elements=[["Al","Mg"]])
        ceBulk = CEBulk(crystalstructure="fcc", a=4.05, size=[3,3,3], concentration=conc, db_name=db_name,  
                        max_cluster_size=3, max_cluster_dia=4.5)
        ceBulk.reconfigure_settings()
        cf = CorrFunction(ceBulk)
        cf = cf.get_cf(ceBulk.atoms)
        ecis = {key:1.0 for key in cf.keys()}
        atoms = ceBulk.atoms.copy()
        calc = CE( atoms, ceBulk, ecis )
        return atoms

    def test_sgc_montecarlo(self):
        if not has_ase_with_ce:
            self.skipTest( "ASE version does not have CE" )
            return

        no_throw = True
        msg = ""
        try:
            atoms = self.get_cebulk()
            T = 600.0
            mc = SGCMonteCarlo(atoms, T, symbols=["Al", "Mg"])
            parameters = [
                {
                    "temperature":10000.0,
                    "chemical_potential":{"c1_0":-1.072}
                },
                {
                    "temperature":9000.0,
                    "chemical_potential":{"c1_0":-1.072}
                }
            ]
            equil_params = {
                "confidence_level":1E-8
            }
            explorer = MCParameterSweep(parameters, mc, nsteps=20, equil_params=equil_params)
            explorer.run()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
