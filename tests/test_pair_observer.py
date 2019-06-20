import unittest
import os
try:
    from cemc.mcmc import PairObserver
    from cemc.mcmc import FixedNucleusMC
    from cemc import CE
    from helper_functions import get_ternary_BC
    from helper_functions import get_example_ecis
    available = True
    skip_msg = ""
except ImportError as exc:
    available = False
    skip_msg = str(exc)

class TestPairObserver(unittest.TestCase):
    def test_pair_observer(self):
        if not available:
            self.skipTest(skip_msg)

        no_throw = True
        msg = ""

        bc = get_ternary_BC()
        ecis = get_example_ecis(bc=bc)
        atoms = bc.atoms.copy()
        calc = CE(atoms, bc, eci=ecis)
        atoms.set_calculator(calc)

        T = 1000
        nn_names = [name for name in bc.cluster_family_names
                    if int(name[1]) == 2]

        mc = FixedNucleusMC(
            atoms, T, network_name=nn_names,
            network_element=["Mg", "Si"])
        mc.insert_symbol_random_places("Mg", swap_symbs=["Al"])
        elements = {"Mg": 3, "Si": 3}
        mc.grow_cluster(elements)
        obs = PairObserver(mc.atoms, cutoff=4.1, elements=["Mg", "Si"])
        mc.attach(obs)
        mc.runMC(steps=200)
        self.assertEqual(obs.num_pairs, obs.num_pairs_brute_force())
        self.assertTrue(obs.symbols_is_synced())


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)