import unittest
import os

try:
    from cemc.mcmc import linear_vib_correction as lvc
    from ase.clease.settings_bulk import CEBulk
    from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
    from cemc.mcmc import Montecarlo
    from cemc import CE, get_ce_calc
    from cemc.mcmc import PairConstraint, FixedElement
    from helper_functions import get_max_cluster_dia_name
    from helper_functions import get_example_network_name
    from helper_functions import get_example_ecis
    from cemc.mcmc.mpi_tools import mpi_communicator
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
        conc_args = {
            "conc_ratio_min_1": [[2, 1, 1]],
            "conc_ratio_max_1": [[0, 2, 2]],
        }
        max_dia_name = get_max_cluster_dia_name()
        size_arg = {max_dia_name: 4.05}
        ceBulk = CEBulk(crystalstructure="fcc", a=4.05, size=[3, 3, 3],
                             basis_elements=[["Al", "Mg", "Si"]],
                             conc_args=conc_args, db_name=db_name,
                             max_cluster_size=3, **size_arg)
        ceBulk.reconfigure_settings()
        calc = CE(ceBulk, ecis)
        ceBulk.atoms.set_calculator(calc)
        return ceBulk

    def test_no_throw(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return
        no_throw = True
        msg = ""
        try:
            ceBulk = self.init_bulk_crystal()
            chem_pots = {
                "c1_0": 0.02,
                "c1_1": -0.03
            }
            T = 600.0
            mc = SGCMonteCarlo(ceBulk.atoms, T, symbols=["Al", "Mg", "Si"])
            mc.runMC(steps=100, chem_potential=chem_pots)
            E = mc.get_thermodynamic()["energy"]

            # Try with recycling
            mc = SGCMonteCarlo(ceBulk.atoms, T, symbols=["Al", "Mg", "Si"],
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

    def test_no_throw_mpi(self):
        if has_ase_with_ce:
            no_throw = True
            msg = ""
            try:
                ceBulk = self.init_bulk_crystal()
                chem_pots = {
                    "c1_0": 0.02,
                    "c1_1": -0.03
                }
                T = 600.0
                comm = mpi_communicator()
                mc = SGCMonteCarlo(
                    ceBulk.atoms, T, symbols=["Al", "Mg", "Si"], mpicomm=comm)
                mc.runMC(steps=100, chem_potential=chem_pots)
                mc.get_thermodynamic()
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
            ceBulk = self.init_bulk_crystal()
            chem_pots = {
                "c1_0": 0.02,
                "c1_1": -0.03
            }
            T = 600.0
            mc = SGCMonteCarlo(ceBulk.atoms, T, symbols=["Al", "Mg", "Si"],
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
            ceBulk = self.init_bulk_crystal()
            chem_pots = {
                "c1_0": 0.02,
                "c1_1": -0.03
            }
            name = get_example_network_name(ceBulk)
            constraint = PairConstraint(
                calc=ceBulk.atoms._calc,
                cluster_name=name,
                elements=[
                    "Al",
                    "Si"])
            # Just an element that is not present to avoid long trials
            fixed_element = FixedElement(element="Cu")
            T = 600.0
            mc = SGCMonteCarlo(
                ceBulk.atoms, T, symbols=[
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
            kwargs = {
            "crystalstructure": "rocksalt",
            "basis_elements": [['V', 'Li'], ['O']],
            "a": 4.12,
            'size': [2, 2, 2],
            'cubic': True,
            "conc_args": {'conc_ratio_min_1': [[1, 1], [2]],
                        'conc_ratio_max_1': [[1, 1], [2]]},
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
            calc = get_ce_calc(ceBulk, kwargs,  eci=ecis, size=[3, 3, 3], 
                            db_name="ignore_test_large.db")
            
            atoms = calc.atoms
            atoms.set_calculator(calc)

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
