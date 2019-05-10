import unittest
import os
import numpy as np

avail_msg = ""
try:
    from cemc.mcmc import PseudoBinarySGC
    from cemc import CE
    from cemc.mcmc import PseudoBinaryFreeEnergyBias
    from cemc.mcmc.reaction_path_utils import PseudoBinaryConcInitializer
    from cemc.mcmc.sgc_montecarlo import InvalidChemicalPotentialError
    from helper_functions import get_ternary_BC, get_example_ecis
    available = True
except ImportError as exc:
    avail_msg = str(exc)
    print(avail_msg)
    available = False


class TestPseudoBinary(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest(avail_msg)

        msg = ""
        try:
            os.remove("test_db_ternary.db")
        except Exception:
            pass
        no_throw = True
        try:
            bc = get_ternary_BC()
            ecis = get_example_ecis(bc=bc)
            atoms = bc.atoms.copy()
            calc = CE(atoms, bc, eci=ecis)

            groups = [{"Al": 2}, {"Mg": 1, "Si": 1}]
            symbs = ["Al", "Mg", "Si"]
            T = 400
            mc = PseudoBinarySGC(atoms, T, chem_pot=-0.2, symbols=symbs,
                                 groups=groups)
            mc.runMC(mode="fixed", steps=100, equil=False)
            os.remove("test_db_ternary.db")
        except Exception as exc:
            msg = "{}: {}".format(type(exc).__name__, str(exc))
            no_throw = False
        self.assertTrue(no_throw, msg)

    def test_with_bias_potential(self):
        if not available:
            self.skipTest(avail_msg)

        msg = ""
        try:
            os.remove("test_db_ternary.db")
        except Exception:
            pass
        no_throw = True
        try:
            bc = get_ternary_BC()
            ecis = get_example_ecis(bc=bc)
            atoms = bc.atoms.copy()
            calc = CE(atoms, bc, eci=ecis)

            groups = [{"Al": 2}, {"Mg": 1, "Si": 1}]
            symbs = ["Al", "Mg", "Si"]
            T = 400
            mc = PseudoBinarySGC(atoms, T, chem_pot=-0.2, symbols=symbs,
                                 groups=groups)
            conc_init = PseudoBinaryConcInitializer(mc)
            reac_crd = np.linspace(0.0, 1.0, 10)
            bias_pot = reac_crd**2
            bias = PseudoBinaryFreeEnergyBias(conc_init, reac_crd,
                                              bias_pot)
            mc.add_bias(bias)
            mc.runMC(mode="fixed", steps=100, equil=False)
            os.remove("test_db_ternary.db")
        except Exception as exc:
            msg = "{}: {}".format(type(exc).__name__, str(exc))
            no_throw = False
        self.assertTrue(no_throw, msg)

    def test_chemical_potential(self):
        if not available:
            self.skipTest(avail_msg)

        bc = get_ternary_BC()
        ecis = {"c1_0": 0.0, "c1_1": 0.0}
        atoms = bc.atoms.copy()
        bf = bc.basis_functions
        calc = CE(atoms, bc, eci=ecis)

        groups = [{"Al": 2}, {"Mg": 1, "Si": 1}]
        symbs = ["Al", "Mg", "Si"]
        T = 400
        chem_pots = [-0.2, -0.1, 0.0, 0.1, 0.2, 1.0]
        all_al = ["Al" for _ in range(len(atoms))]
        num_atom_per_fu = 2
        for mu in chem_pots:
            calc.set_symbols(all_al)
            mc = PseudoBinarySGC(atoms, T, chem_pot=mu,
                                 symbols=symbs, groups=groups)

            # At this point the chemical potentials should
            # be updated
            changes = [(0, "Al", "Mg"), (1, "Al", "Si")]
            energy = calc.get_energy()
            for change in changes:
                calc.update_cf(change)

            new_energy = calc.get_energy()
            self.assertAlmostEqual(new_energy - energy, mu)
            mc.reset_ecis()

    def test_raise_error_on_invalid_chem_pot(self):
        if not available:
            self.skipTest(avail_msg)

        bc = get_ternary_BC()
        ecis = get_example_ecis(bc=bc)
        ecis.pop('c1_1')
        atoms = bc.atoms.copy()
        calc = CE(atoms, bc, eci=ecis)
        atoms.set_calculator(calc)

        groups = [{"Al": 2}, {"Mg": 1, "Si": 1}]
        symbs = ["Al", "Mg", "Si"]
        T = 400
        with self.assertRaises(InvalidChemicalPotentialError):
            PseudoBinarySGC(atoms, T, chem_pot=-0.2, symbols=symbs,
                            groups=groups)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
