import unittest
try:
    from ase.ce.settings_bulk import BulkCrystal
    from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
    from cemc.wanglandau.ce_calculator import CE
    from cemc.mcmc.mc_parameter_sweep import MCParameterSweep
    has_ase_with_ce = True
except Exception as exc:
    print (str(exc))
    has_ase_with_ce = False

ecis = {
    "c1_1":-0.1,
    "c2_1000_1_1":0.1,
}

db_name = "test_sgc.db"
class TestMCParameterSweep( unittest.TestCase ):
    def get_cebulk(self):
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        ceBulk = BulkCrystal( "fcc", 4.05, None, [3,3,3], 1, [["Al","Mg"]], conc_args, db_name, reconf_db=False)
        calc = CE( ceBulk, ecis )
        ceBulk.atoms.set_calculator(calc)
        return ceBulk

    def test_sgc_montecarlo(self):
        if ( not has_ase_with_ce ):
            self.skipTest( "ASE version does not have CE" )
            return

        no_throw = True
        try:
            bc = self.get_cebulk()
            T = 600.0
            mc = SGCMonteCarlo( bc.atoms, T, symbols=["Al","Mg"] )
            parameters = [
                {
                    "temperature":10000.0,
                    "chemical_potential":{"c1_1":-1.072}
                },
                {
                    "temperature":9000.0,
                    "chemical_potential":{"c1_1":-1.072}
                }
            ]
            equil_params = {
                "confidence_level":1E-8
            }
            explorer = MCParameterSweep( parameters, mc, nsteps=20, equil_params=equil_params )
            explorer.run()
        except Exception as exc:
            print(exc)
            no_throw = False
        self.assertTrue( no_throw )

if __name__ == "__main__":
    unittest.main()
