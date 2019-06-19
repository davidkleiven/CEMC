import unittest
import os
try:
    has_CE = True
    from ase.clease import CEBulk, CorrFunction, Concentration
    from cemc import get_atoms_with_ce_calc
    from cemc.mcmc import SGCFreeEnergyBarrier
except ImportError as exc:
    print (str(exc))
    has_CE = False

class TestFreeEnergy( unittest.TestCase ):
    def test_no_throw(self):
        if ( not has_CE ):
            self.skipTest( "ASE version does not have ASE" )
            return
        no_throw = True
        msg = ""
        try:
            conc_args = {
                        "conc_ratio_min_1":[[1,0]],
                        "conc_ratio_max_1":[[0,1]],
                    }
            conc = Concentration(basis_elements=[["Al","Mg"]])
            kwargs = {
                "crystalstructure":"fcc", "a":4.05, "size":[3, 3, 3],
                "concentration": conc, "db_name":"temporary_bcnucleationdb.db",
                "max_cluster_size": 3, "max_cluster_dia": 4.5
            }
            ceBulk = CEBulk( **kwargs )
            cf = CorrFunction(ceBulk)
            cf_dict = cf.get_cf(ceBulk.atoms)
            ecis = {key:1.0 for key,value in cf_dict.items()}


            #calc = CE( ceBulk, ecis, size=(3,3,3) )
            atoms = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[6,6,6],
                               db_name="sc6x6x6.db")
            chem_pot = {"c1_0":-1.069}

            T = 300
            mc = SGCFreeEnergyBarrier( atoms, T, symbols=["Al","Mg"], \
            n_windows=5, n_bins=10, min_singlet=0.5, max_singlet=1.0 )
            mc.run( nsteps=100, chem_pot=chem_pot )
            mc.save( fname="free_energy_barrier.json" )

            # Try to load the object stored
            mc = SGCFreeEnergyBarrier.load( atoms, "free_energy_barrier.json")
            mc.run( nsteps=100, chem_pot=chem_pot )
            mc.save( fname="free_energy_barrier.json" )
            os.remove("sc6x6x6.db")
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
