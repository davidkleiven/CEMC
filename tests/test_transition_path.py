import unittest

try:
    from cemc.mcmc import NucleationSampler, SGCNucleation
    from cemc.mcmc.sgc_nucleation_mc import DidNotReachProductOrReactantError
    from cemc.mcmc.sgc_nucleation_mc import DidNotFindPathError
    from cemc.mcmc import TransitionPathRelaxer
    from ase.ce import BulkCrystal
    from cemc import get_ce_calc
    from helper_functions import get_small_BC_with_ce_calc, get_example_network_name
    available = True
    available_reason = ""
except Exception as exc:
    available_reason = str(exc)
    print (available_reason)
    available = False


class TestTransitionPath(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest("Transition path test skipped because: "+available_reason)
            return

        no_throw = True
        msg = ""
        try:
            ceBulk = get_small_BC_with_ce_calc()
            chem_pot = {"c1_0":-1.065}
            sampler = NucleationSampler( size_window_width=10, \
            chemical_potential=chem_pot, max_cluster_size=150, \
            merge_strategy="normalize_overlap", max_one_cluster=False )

            T = 500
            network_name = get_example_network_name(ceBulk)
            mc = SGCNucleation( ceBulk.atoms, T, nucleation_sampler=sampler, \
            network_name=network_name,  network_element="Mg", symbols=["Al","Mg"], \
            chem_pot=chem_pot, allow_solutes=True )

            try:
                mc.find_transition_path( initial_cluster_size=5, max_size_reactant=10, min_size_product=15, \
                folder="data", path_length=4, max_attempts=1, nsteps=int(0.1*len(ceBulk.atoms)) )
            except (DidNotReachProductOrReactantError, DidNotFindPathError):
                pass

            relaxer = TransitionPathRelaxer( nuc_mc=mc )
            relaxer.relax_path( initial_path="tests/test_data/example_path.json", n_shooting_moves=1 )
            relaxer.path2trajectory(fname="data/relaxed_path.traj")

            relaxer = TransitionPathRelaxer( nuc_mc=mc )
            relaxer.generate_paths( initial_path="tests/test_data/example_path.json", n_paths=1, max_attempts=2, outfile="data/tse_ensemble.json" )
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
