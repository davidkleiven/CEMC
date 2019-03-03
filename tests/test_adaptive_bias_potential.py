import unittest
try:
    from cemc.mcmc import MCObserver
    from cemc.mcmc import SGCMonteCarlo
    from cemc.mcmc import AdaptiveBiasReactionPathSampler
    from cemc.mcmc import ReactionCrdRangeConstraint
    from cemc import CE
    from helper_functions import get_ternary_BC
    from helper_functions import get_example_ecis
    from ase.units import kB
    import numpy as np
    from scipy.special import factorial
    from itertools import product
    import os

    class DummyObserver(MCObserver):
        def __init__(self, atoms):
            self.atoms = atoms
            self.num_atoms = len(atoms)
            self.al_conc = self.calculate_from_scratch(atoms)["value"]

        def __call__(self, system_changes, peak=False):

            num_new_al = 0
            for change in system_changes:
                if change[2] == "Al":
                    num_new_al += 1.0
                if change[1] == "Al":
                    num_new_al -= 1.0
            old_conc = self.al_conc
            self.al_conc += num_new_al/self.num_atoms
            cur_val = self.get_current_value()

            assert self.al_conc >= 0.0 and self.al_conc <= 1.0, \
                "{}: {} {}".format(system_changes, self.al_conc,
                                   self.atoms.get_chemical_formula())

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
    def get_num_configurations(self, num_al, tot_num_atoms):
        num_config = 0
        for num_mg_si in product(range(tot_num_atoms), repeat=2):
            if sum(num_mg_si) + num_al != tot_num_atoms:
                continue
            num_config += factorial(tot_num_atoms)/(factorial(num_mg_si[0])*factorial(num_mg_si[1])*factorial(num_al))
        return num_config

    def test_no_throw(self):
        if not available:
            self.skipTest(skipMsg)

        no_throw = True
        msg = ""

        bc = get_ternary_BC(N=2, max_size=2, db_name="ideal.db")
        eci = {"c1_0": 0.0, "c1_1": 0.0}

        atoms = bc.atoms.copy()
        CE(atoms, bc, eci=eci)
        obs = DummyObserver(atoms)
        cnst = ReactionCrdRangeConstraint(obs, value_name="value")

        

        # Note: convergence_factor should never be
        # set to a negative value! Here it is done
        # to make the test converged after one step
        mod_factor = 0.01
        while mod_factor > 1E-8:
            mc = SGCMonteCarlo(atoms, 1000, symbols=["Al", "Mg", "Si"])
            mc.chemical_potential = {"c1_0": 0.0, "c1_1": 0.0}
            print("Mod. factor: {}".format(mod_factor))
            bias = AdaptiveBiasReactionPathSampler(
                mc_obj=mc, observer=obs,
                react_crd=[0.0, 1.01], convergence_factor=0.8,
                react_crd_name="value", n_bins=9, mod_factor=mod_factor,
                data_file="ideal_mixture.h5", db_struct="ideal_mixt.db")
            bias.run()
            bias.save()
            mod_factor /= 2.0

        # Expected result
        os.remove("ideal_mixt.db")
        os.remove("ideal_mixture.h5")
        num_al = list(range(9))
        expected_num_config = [self.get_num_configurations(x, 8) for x in num_al]
        expected_num_config = np.log(expected_num_config)
        array = bias.bias.bias_array.copy()
        array -= array[-1]
        print(expected_num_config, array)
        self.assertTrue(no_throw, msg=msg)

    def test_continuous_curve(self):
        if not available:
            self.skipTest(skipMsg)
        bc = get_ternary_BC()
        eci = {"c1_0": 0.0, "c1_1": 0.0}

        atoms = bc.atoms.copy()
        CE(atoms, bc, eci=eci)
        obs = DummyObserver(atoms)
        cnst = ReactionCrdRangeConstraint(obs, value_name="value")

        T = 1000.0
        mc = SGCMonteCarlo(atoms, T, symbols=["Al", "Mg", "Si"])
        mc.add_constraint(cnst)
        mc.chemical_potential = {"c1_0": 0.0, "c1_1": 0.0}

        self.assertAlmostEqual(mc.current_energy, 0.0)
        self.assertAlmostEqual(obs.al_conc, 1.0)
        num_mg = int(len(atoms)/2)
        mc.insert_symbol_random_places("Mg", num=num_mg, swap_symbs=["Al"])
        obs.calculate_from_scratch(atoms)
        self.assertAlmostEqual(obs.al_conc, 0.5)

        bias = AdaptiveBiasReactionPathSampler(
            mc_obj=mc, observer=obs,
            react_crd=[0.0, 1.0], convergence_factor=-1.0,
            react_crd_name="value", n_bins=32)
        
        bias.current_min_bin = 30
        bias.connection = {"bin": 30, "value": 10.0}
        bias._make_energy_curve_continuous()

        # The original value was 0, but above we have
        # specified that bin 30 and higher should have
        # a value 10. Hence, the energy should not change
        self.assertAlmostEqual(mc.current_energy, 0.0)

        bias.current_min_bin = 2
        bias.connection = {"bin": 2, "value": 5.0}

        # In this case the total energy should be increased by 5
        bias._make_energy_curve_continuous()
        self.assertAlmostEqual(mc.current_energy, 5.0*kB*T)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
