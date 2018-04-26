import unittest
try:
    from cemc.mcmc import NucleationMC
    from ase.ce import BulkCrystal
    from cemc.wanglandau.ce_calculator import get_ce_calc
    available = True
except Exception as exc:
    print (str(exc))
    available = False

class TestNuclFreeEnergy( unittest.TestCase ):
    def test_no_throw(self):
        if ( not available ):
            self.skipTest( "ASE version does not have CE!" )
            return
        no_throw = True
        msg = ""
        try:
            conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
            kwargs = {
                "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
                "conc_args":conc_args, "db_name":"data/temporary_bcnucleationdb.db",
                "max_cluster_size":4
            }
            ceBulk = BulkCrystal( **kwargs )
            print (ceBulk.basis_functions)

            ecis = {"c1_0":-0.01,"c2_1414_1_00":-0.2}
            calc = get_ce_calc( ceBulk, kwargs, ecis, size=[5,5,5], free_unused_arrays_BC=True )
            ceBulk = calc.BC
            ceBulk.atoms.set_calculator( calc )

            chem_pot = {"c1_0":-1.0651526881167124}
            mc = NucleationMC( ceBulk.atoms, 300, size_window_width=5, network_name="c2_1414_1", network_element="Mg", symbols=["Al","Mg"], \
            chemical_potential=chem_pot, max_cluster_size=10, merge_strategy="normalize_overlap" )
            mc.run(nsteps=2)
            mc.save(fname="test_nucl.h5")
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw,msg=msg)

if __name__ == "__main__":
    unittest.main()
