import unittest
import os

try:
    from cemc.mcmc import linear_vib_correction as lvc
    from ase.clease.settings_bulk import CEBulk
    from ase.clease import Concentration
    from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo, InvalidChemicalPotentialError
    from cemc.mcmc import Montecarlo
    from cemc import CE, get_atoms_with_ce_calc
    from cemc.mcmc import PairConstraint, FixedElement
    from helper_functions import get_max_cluster_dia_name
    from helper_functions import get_example_network_name
    from helper_functions import get_example_ecis
    has_ase_with_ce = True
except Exception as exc:
    print(str(exc))
    has_ase_with_ce = False

ecis = {
    "c1_0": -0.1,
    "c1_1": 0.1,
}

db_name = "test_sgc.db"


class TestSGCMC(unittest.TestCase):
    def init_bulk_crystal(self):
        max_dia_name = get_max_cluster_dia_name()
        size_arg = {max_dia_name: 4.05}
        conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])
        ceBulk = CEBulk(crystalstructure="fcc", a=4.05, size=[3, 3, 3],
                             concentration=conc, db_name=db_name,
                             max_cluster_size=3, **size_arg)
        ceBulk.reconfigure_settings()
        atoms = ceBulk.atoms.copy()
        calc = CE(atoms, ceBulk, ecis)
        return ceBulk, atoms

    def test_no_throw(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return
        no_throw = True
        msg = ""
        try:
            ceBulk, atoms = self.init_bulk_crystal()
            chem_pots = {
                "c1_0": 0.02,
                "c1_1": -0.03
            }
            T = 600.0
            mc = SGCMonteCarlo(atoms, T, symbols=["Al", "Mg", "Si"])
            mc.runMC(steps=100, chem_potential=chem_pots)
            E = mc.get_thermodynamic()["energy"]

            # Try with recycling
            mc = SGCMonteCarlo(atoms, T, symbols=["Al", "Mg", "Si"],
                               recycle_waste=True)
            mc.runMC(steps=100, chem_potential=chem_pots)
            E2 = mc.get_thermodynamic()["energy"]
            rel_diff = abs(E2 - E)/abs(E)
            # Make sure that there is less than 10% difference to rule
            # out obvious bugs
            self.assertLess(rel_diff, 0.1)

            # Try to pickle
            import dill
            dill.pickles(mc)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

    def test_no_throw_prec_mode(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return
        no_throw = True
        msg = ""
        try:
            ceBulk, atoms = self.init_bulk_crystal()
            chem_pots = {
                "c1_0": 0.02,
                "c1_1": -0.03
            }
            T = 600.0
            mc = SGCMonteCarlo(atoms, T, symbols=["Al", "Mg", "Si"],
                               plot_debug=False)
            mc.runMC(chem_potential=chem_pots, mode="prec",
                     prec_confidence=0.05, prec=10.0)
            mc.get_thermodynamic()

            eci_vib = {"c1_0": 0.0}
            vib_corr = lvc.LinearVibCorrection(eci_vib)
            mc.linear_vib_correction = vib_corr
            mc.runMC(chem_potential=chem_pots, mode="prec",
                     prec_confidence=0.05, prec=10.0)
            mc.get_thermodynamic()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

    def test_constraints(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return
        no_throw = True
        msg = ""
        try:
            ceBulk, atoms = self.init_bulk_crystal()
            chem_pots = {
                "c1_0": 0.02,
                "c1_1": -0.03
            }
            name = get_example_network_name(ceBulk)
            constraint = PairConstraint(
                calc=atoms.get_calculator(),
                cluster_name=name,
                elements=[
                    "Al",
                    "Si"])
            # Just an element that is not present to avoid long trials
            fixed_element = FixedElement(element="Cu")
            T = 600.0
            mc = SGCMonteCarlo(
                atoms, T, symbols=[
                    "Al", "Mg", "Si"], plot_debug=False)
            mc.add_constraint(constraint)
            mc.add_constraint(fixed_element)
            mc.runMC(chem_potential=chem_pots, mode="fixed", steps=10)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

    
    def test_ignore_atoms(self):
        if not has_ase_with_ce:
            self.skipTest("ASE does not have CE")
        from cemc.mcmc import FixedElement
        no_trow = True
        msg = ""
        try:
            from copy import deepcopy
            conc = Concentration(basis_elements=[['V', 'Li'], ['O']])
            kwargs = {
            "crystalstructure": "rocksalt",
            "concentration": conc,
            "a": 4.12,
            'size': [2, 2, 2],
            'cubic': True,
            "max_cluster_size": 4,
            "max_cluster_dia": 4.12,
            "db_name": 'database.db',
            'basis_function': 'sluiter',
            'ignore_background_atoms': True
            }
            fix_elem = FixedElement(element="O")
            kw_args_cpy = deepcopy(kwargs)
            ceBulk = CEBulk(**kw_args_cpy)
            ecis = get_example_ecis(ceBulk)
            atoms = get_atoms_with_ce_calc(ceBulk, kwargs,  eci=ecis, size=[3, 3, 3], 
                            db_name="ignore_test_large.db")
            calc = atoms.get_calculator()

            # Insert some Li atoms
            num_li = 5
            symbols = [atom.symbol for atom in atoms]
            num_inserted = 0
            for i in range(0, len(symbols)):
                if symbols[i] == "V":
                    symbols[i] = "Li"
                    num_inserted += 1
                if num_inserted >= num_li:
                    break
            calc.set_symbols(symbols)

            mc = Montecarlo(atoms, 800)
            mc.add_constraint(fix_elem)
            mc.runMC(steps=100, equil=False, mode="fixed")

            # Clean up files
            os.remove("ignore_test_large.db")
            os.remove("database.db")
        except Exception as exc:
            no_trow = False
            msg = str(exc)
        self.assertTrue(no_trow, msg=msg)


    def __del__(self):
        if (os.path.isfile(db_name)):
            os.remove(db_name)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
