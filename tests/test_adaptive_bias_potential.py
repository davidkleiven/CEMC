import unittest
try:
    from cemc.mcmc import ReactionCrdInitializer
    from cemc.mcmc import Montecarlo
    from cemc.mcmc import AdaptiveBiasReactionPathSampler
    from cemc import CE
    from helper_functions import get_ternary_BC
    from helper_functions import get_example_ecis
    import numpy as np

    class DummyReacCrdInit(ReactionCrdInitializer):
        def get(self, atoms, system_changes=[]):
            return np.random.rand()*2.0 - 1.0

        def set(self, value):
            pass

        def __call__(self, system_changes):
            return 0.0

    skipMsg = ""
    available = True
except ImportError as exc:
    skipMsg = str(exc)
    available = False


class TestAdaptiveBiasReacPath(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest(skipMsg)

        no_throw = True
        msg = ""
        try:
            bc = get_ternary_BC()
            eci = get_example_ecis(bc)

            atoms = bc.atoms.copy()
            CE(atoms, bc, eci=eci)

            mc = Montecarlo(atoms, 1000)
            mc.insert_symbol_random_places("Mg", num=10, swap_symbs=["Al"])

            # Note: convergence_factor should never be
            # set to a negative value! Here it is done
            # to make the test converged after one step
            bias = AdaptiveBiasReactionPathSampler(
                mc_obj=mc, react_crd_init=DummyReacCrdInit(), 
                react_crd=[-1.0, 1.0], convergence_factor=-1.0)
            bias.run()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
