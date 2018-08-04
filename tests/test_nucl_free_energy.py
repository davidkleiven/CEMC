import unittest
try:
    from cemc.mcmc import NucleationSampler,SGCNucleation, CanonicalNucleationMC, FixedNucleusMC
    from ase.ce import BulkCrystal
    from ase.ce import CorrFunction
    from cemc import get_ce_calc
    from helper_functions import flatten_cluster_names
    import os
    available = True
except Exception as exc:
    print (str(exc))
    available = False

def get_network_name(cnames):
    for name in cnames:
        if int(name[1]) == 2:
            return name
    raise RuntimeError("No pair cluster found!")

class TestNuclFreeEnergy( unittest.TestCase ):
    def test_no_throw(self):
        if ( not available ):
            self.skipTest( "ASE version does not have CE!" )
            return
        no_throw = True
        msg = ""
        try:
            conc_args = {
                "conc_ratio_min_1": [[1, 0]],
                "conc_ratio_max_1": [[0, 1]],
            }
            db_name = "temp_nuc_db.db"
            if os.path.exists(db_name):
                os.remove(db_name)
            kwargs = {
                "crystalstructure":"fcc", "a": 4.05,
                "size":[3, 3, 3],
                "basis_elements":[["Al", "Mg"]],
                "conc_args": conc_args, "db_name": db_name,
                "max_cluster_size": 3
            }
            ceBulk = BulkCrystal(**kwargs)
            ceBulk.reconfigure_settings()
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)

            ecis = {key: 0.001 for key in cf.keys()}
            calc = get_ce_calc(ceBulk, kwargs, ecis, size=[5, 5, 5])
            ceBulk = calc.BC
            ceBulk.atoms.set_calculator(calc)

            chem_pot = {"c1_0": -1.0651526881167124}
            sampler = NucleationSampler(
                size_window_width=10,
                chemical_potential=chem_pot, max_cluster_size=20,
                merge_strategy="normalize_overlap")

            nn_name = get_network_name(ceBulk.cluster_family_names_by_size)

            mc = SGCNucleation(
                ceBulk.atoms, 300, nucleation_sampler=sampler,
                network_name=nn_name,  network_element="Mg",
                symbols=["Al", "Mg"], chem_pot=chem_pot)

            mc.run(nsteps=2)
            sampler.save(fname="test_nucl.h5")

            mc = CanonicalNucleationMC(
                ceBulk.atoms, 300, nucleation_sampler=sampler,
                network_name=nn_name,  network_element="Mg",
                concentration={"Al": 0.8, "Mg": 0.2}
                )
            mc.run(nsteps=2)
            sampler.save(fname="test_nucl_canonical.h5")

            mc = FixedNucleusMC(ceBulk.atoms, 300, size=6,
                                network_name=nn_name, network_element="Mg")
            mc.run(nsteps=2)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
