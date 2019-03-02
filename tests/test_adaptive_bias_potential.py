import unittest
try:
    from cemc.mcmc import MCObserver
    from cemc.mcmc import SGCMonteCarlo
    from cemc.mcmc import AdaptiveBiasReactionPathSampler
    from cemc.mcmc import ReactionCrdRangeConstraint
    from cemc import CE
    from helper_functions import get_ternary_BC
    from helper_functions import get_example_ecis
    import numpy as np

    class DummyObserver(MCObserver):
        def __init__(self, atoms):
            self.num_atoms = len(atoms)
            self.al_conc = self.calculate_from_scratch(atoms)["value"]

        def __call__(self, system_changes, peak=False):

            num_new_al = 0
            for change in system_changes:
                if change[2] == "Al":
                    num_new_al += 1
                if change[1] == "Al":
                    num_new_al -= 1
            old_conc = self.al_conc
            self.al_conc += num_new_al/self.num_atoms
            cur_val = self.get_current_value()

            if peak:
                self.al_conc = old_conc
            return cur_val

        def calculate_from_scratch(self, atoms):
            assert len(atoms) == self.num_atoms
            num_al = sum(1 for atom in atoms if atom.symbol == "Al")
            self.al_conc = float(num_al)/self.num_atoms
            return self.get_current_value()

        def get_current_value(self):
            return {"value": self.al_conc}

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

        bc = get_ternary_BC()
        eci = {"c1_0": 0.0, "c1_1": 0.0}

        atoms = bc.atoms.copy()
        CE(atoms, bc, eci=eci)
        obs = DummyObserver(atoms)
        cnst = ReactionCrdRangeConstraint(obs, value_name="value")

        mc = SGCMonteCarlo(atoms, 1000, symbols=["Al", "Mg", "Si"])
        mc.add_constraint(cnst)
        mc.chemical_potential = {"c1_0": 0.0, "c1_1": 0.0}
        mc.insert_symbol_random_places("Mg", num=10, swap_symbs=["Al"])

        # Note: convergence_factor should never be
        # set to a negative value! Here it is done
        # to make the test converged after one step
        bias = AdaptiveBiasReactionPathSampler(
            mc_obj=mc, observer=obs,
            react_crd=[0.0, 1.0], convergence_factor=0.1,
            react_crd_name="value", n_bins=32)
        bias.run()

        self.assertTrue(no_throw, msg=msg)

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
