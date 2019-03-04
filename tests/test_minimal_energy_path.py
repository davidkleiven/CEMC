import unittest
import numpy as np

try:
    from cemc.mcmc import MinimalEnergyPath
    from cemc.mcmc import SGCMonteCarlo
    from cemc.mcmc import MCObserver
    from helper_functions import get_binary_BC
    from cemc import CE
    import os
    available = True
    reason = ""

    class ConcObserver(MCObserver):
        def __init__(self, atoms, symbol):
            MCObserver.__init__(self)
            self.symbol = symbol
            self.atoms = atoms
            self.cur_val = self.calculate_from_scratch(self.atoms)

        def calculate_from_scratch(self, atoms):
            num_al = sum(1 for atom in atoms if atom.symbol == self.symbol)
            return float(num_al)/len(atoms)

        def get_current_value(self):
            return {"symb_conc": self.cur_val}

        def __call__(self, system_changes, peak=False):
            num_new = 0
            for change in system_changes:
                if change[2] == self.symbol:
                    num_new += 1
                if change[1] == self.symbol:
                    num_new -= 1

            old_val = self.cur_val
            self.cur_val += float(num_new)/len(self.atoms)
            cur_val = self.get_current_value()
            if peak:
                self.cur_val = old_val
            return cur_val


except ImportError as exc:
    available = False
    reason = str(exc)


class TestMinimalEnergyPath(unittest.TestCase):
    def test_ideal_mixture(self):
        if not available:
            self.skipTest(reason)

        bc = get_binary_BC(db_name="binary.db", N=3)
        atoms = bc.atoms.copy()
        eci = {"c1_0": 0.1}

        calc = CE(atoms, bc, eci=eci)
        atoms.set_calculator(calc)

        mc = SGCMonteCarlo(atoms, 10, symbols=["Al", "Mg"])
        mc.chemical_potential = {"c1_0": 0.0}

        conc_observer = ConcObserver(atoms, "Mg")
        cur_val = conc_observer.get_current_value()["symb_conc"]
        self.assertAlmostEqual(cur_val, 0.0)
        me_path = MinimalEnergyPath(mc_obj=mc, observer=conc_observer,
                                    value_name="symb_conc", relax_steps=1000,
                                    search_steps=1000,
                                    traj_file="test_mep.traj",
                                    max_reac_crd=0.99)

        me_path.run()
        N = len(atoms)
        expected = [eci["c1_0"]*(N-2.0*n)
                    for n in range(N)]
        os.remove("test_mep.traj")
        os.remove("binary.db")
        self.assertTrue(np.allclose(expected, me_path.energy))


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)