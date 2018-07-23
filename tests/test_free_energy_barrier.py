import unittest
try:
    has_CE = True
    from ase.ce import BulkCrystal, CorrFunction
    from cemc import get_ce_calc
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
            kwargs = {
                "crystalstructure":"fcc", "a":4.05, "size":[3, 3, 3], "basis_elements":[["Al","Mg"]],
                "conc_args":conc_args, "db_name":"temporary_bcnucleationdb.db",
                "max_cluster_size": 3
            }
            ceBulk = BulkCrystal( **kwargs )
            cf = CorrFunction(ceBulk)
            cf_dict = cf.get_cf(ceBulk.atoms)
            ecis = {key:1.0 for key,value in cf_dict.iteritems()}


            #calc = CE( ceBulk, ecis, size=(3,3,3) )
            calc = get_ce_calc( ceBulk, kwargs, ecis, size=[6,6,6], free_unused_arrays_BC=True )
            ceBulk = calc.BC
            ceBulk.atoms.set_calculator( calc )
            chem_pot = {"c1_0":-1.069}

            T = 300
            mc = SGCFreeEnergyBarrier( ceBulk.atoms, T, symbols=["Al","Mg"], \
            n_windows=5, n_bins=10, min_singlet=0.5, max_singlet=1.0 )
            mc.run( nsteps=100, chem_pot=chem_pot )
            mc.save( fname="free_energy_barrier.json" )

            # Try to load the object stored
            mc = SGCFreeEnergyBarrier.load( ceBulk.atoms, "free_energy_barrier.json")
            mc.run( nsteps=100, chem_pot=chem_pot )
            mc.save( fname="free_energy_barrier.json" )
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
