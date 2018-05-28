import unittest
try:
    from cemc.mcmc import NucleationSampler,SGCNucleation, CanonicalNucleationMC, FixedNucleusMC
    from ase.ce import BulkCrystal
    from ase.ce import CorrFunction
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
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)

            ecis = {key:0.001 for key in cf.keys()}
            calc = get_ce_calc( ceBulk, kwargs, ecis, size=[5,5,5], free_unused_arrays_BC=False )
            ceBulk = calc.BC
            ceBulk.atoms.set_calculator( calc )

            chem_pot = {"c1_0":-1.0651526881167124}
            sampler = NucleationSampler( size_window_width=10, \
            chemical_potential=chem_pot, max_cluster_size=20, \
            merge_strategy="normalize_overlap" )

            mc = SGCNucleation( ceBulk.atoms, 300, nucleation_sampler=sampler, \
            network_name="c2_1414_1",  network_element="Mg", symbols=["Al","Mg"], \
            chem_pot=chem_pot )
            mc.run(nsteps=2)
            sampler.save(fname="test_nucl.h5")

            mc = CanonicalNucleationMC( ceBulk.atoms, 300, nucleation_sampler=sampler, \
            network_name="c2_1414_1",  network_element="Mg", \
            concentration={"Al":0.8,"Mg":0.2} )
            mc.run(nsteps=2)
            sampler.save(fname="test_nucl_canonical.h5")

            mc = FixedNucleusMC( ceBulk.atoms, 300, size=6, network_name="c2_1414_1", network_element="Mg" )
            mc.run(nsteps=2)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw,msg=msg)

if __name__ == "__main__":
    unittest.main()
