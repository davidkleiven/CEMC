import unittest
try:
    from cemc.tools import MultithreadPerformance
    from cemc.mcmc import Montecarlo
    from helper_functions import get_ternary_BC, get_example_ecis
    from cemc import CE
    reason = ""
    available = True
except ImportError as exc:
    reason = str(exc)
    print(reason)
    available = False


class TestMultithread(unittest.TestCase):
    def test_run(self):
        if not available:
            self.skipTest(reason)

        bc = get_ternary_BC()
        eci = get_example_ecis(bc)
        atoms = bc.atoms.copy()
        calc = CE(atoms, bc, eci=eci)

        mc = Montecarlo(atoms, 1000000)
        mc.insert_symbol_random_places("Mg", swap_symbs=["Al"], num=3)
        performance_monitor = MultithreadPerformance(4)
        performance_monitor.run(mc, 100)


if __name__ == "__main__":
    unittest.main()
