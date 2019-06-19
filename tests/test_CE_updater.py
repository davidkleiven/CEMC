import unittest
import os
try:
    from cemc import TimeLoggingTestRunner
    from ase.clease.settings_bulk import CEBulk
    from ase.clease import Concentration
    from ase.clease.corrFunc import CorrFunction
    from cemc import CE, get_atoms_with_ce_calc
    from cemc import get_atoms_with_ce_calc_JSON
    from helper_functions import get_bulkspacegroup_binary
    from helper_functions import get_max_cluster_dia_name
    import copy
    has_ase_with_ce = True
except Exception as exc:
    print(str(exc))
    has_ase_with_ce = False
import numpy as np


class TestCE(unittest.TestCase):
    lattices = ["fcc", "bcc", "sc", "hcp"]

    def get_calc(self, lat):
        db_name = "test_db_{}.db".format(lat)

        conc = Concentration(basis_elements=[["Al", "Mg"]])
        a = 4.05
        ceBulk = CEBulk(crystalstructure=lat, a=a, size=[3, 3, 3],
                        concentration=conc, db_name=db_name,
                        max_cluster_size=3, max_cluster_dia=6.0)
        ceBulk.reconfigure_settings()
        cf = CorrFunction(ceBulk)
        corrfuncs = cf.get_cf(ceBulk.atoms)
        eci = {name: 1.0 for name in corrfuncs.keys()}
        atoms = ceBulk.atoms.copy()
        calc = CE(atoms, ceBulk, eci)
        atoms.set_calculator(calc)
        return atoms, ceBulk, eci

    def test_update(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
        for lat in self.lattices:
            msg = "Failed for lattice {}".format(lat)
            atoms, ceBulk, _ = self.get_calc(lat)
            calc = atoms.get_calculator()
            cf = CorrFunction(ceBulk)
            n_tests = 10
            for i in range(n_tests):
                old_symb = atoms[i].symbol
                if old_symb == "Al":
                    new_symb = "Mg"
                else:
                    new_symb = "Al"
                calc.update_cf((i, old_symb, new_symb))
                updated_cf = calc.get_cf()
                brute_force = cf.get_cf(atoms)
                for key, value in updated_cf.items():
                    self.assertAlmostEqual(value, brute_force[key], msg=msg)

    def test_random_swaps(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return

        for lat in self.lattices:
            msg = "Failed for lattice {}".format(lat)
            atoms, ceBulk, _ = self.get_calc(lat)
            n_tests = 10
            corr_func = CorrFunction(ceBulk)
            calc = atoms.get_calculator()
            for _ in range(n_tests):
                indx = np.random.randint(low=0, high=len(ceBulk.atoms))
                old_symb = atoms[indx].symbol
                if old_symb == "Al":
                    new_symb = "Mg"
                else:
                    new_symb = "Al"
                calc.calculate(atoms, ["energy"],
                               [(indx, old_symb, new_symb)])
                updated_cf = calc.get_cf()
                brute_force = corr_func.get_cf_by_cluster_names(atoms, updated_cf.keys())
                for key, value in updated_cf.items():
                    self.assertAlmostEqual(value, brute_force[key], msg=msg)


    def test_double_swaps(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return

        for lat in self.lattices:
            atoms, ceBulk, _ = self.get_calc(lat)
            calc = atoms.get_calculator()
            corr_func = CorrFunction(ceBulk)
            n_tests = 10
            symbs = [atom.symbol for atom in atoms]
            # Insert 10 Mg atoms
            for i in range(10):
                symbs[i] = "Mg"
            calc.set_symbols(symbs)

            # Swap Al and Mg atoms
            for i in range(n_tests):
                indx1 = np.random.randint(low=0, high=len(ceBulk.atoms))
                symb1 = atoms[indx1].symbol
                indx2 = indx1
                symb2 = symb1
                while symb2 == symb1:
                    indx2 = np.random.randint(low=0, high=len(atoms))
                    symb2 = atoms[indx2].symbol
                calc.calculate(atoms, ["energy"], [(indx1, symb1, symb2),(indx2, symb2, symb1)])
                updated_cf = calc.get_cf()
                brute_force = corr_func.get_cf_by_cluster_names(atoms, updated_cf.keys())
                for key,value in brute_force.items():
                    self.assertAlmostEqual(value, updated_cf[key])

    def test_supercell(self):
        if not has_ase_with_ce:
            self.skipTest("ASE version does not have CE")
            return

        atoms, ceBulk, eci = self.get_calc("fcc")
        calc = atoms.get_calculator()
        db_name = "test_db_fcc_super.db"

        conc = Concentration(basis_elements=[["Al", "Mg"]])
        kwargs = {
            "crystalstructure": "fcc",
            "a": 4.05,
            "size":[3, 3, 3],
            "concentration": conc,
            "db_name": db_name,
            "max_cluster_size": 3,
            "max_cluster_dia": 6.0
        }

        kwargs_template = copy.deepcopy(kwargs)
        kwargs_template["size"] = [4, 4, 4]
        kwargs_template["db_name"] = "template_bc.db"
        template_supercell_bc = CEBulk(**kwargs_template)
        template_supercell_bc.reconfigure_settings()

        ceBulk = CEBulk(**kwargs)
        ceBulk.reconfigure_settings()
        atoms_sc = get_atoms_with_ce_calc(ceBulk, kwargs, eci, size=[4, 4, 4],
                                          db_name="sc4x4x.db")
        calc_sc = atoms_sc.get_calculator()
        corr_func = CorrFunction(template_supercell_bc)
        for i in range(10):
            calc_sc.calculate(atoms_sc, ["energy"], [(i, "Al", "Mg")])
            updated_cf = calc_sc.get_cf()
            brute_force = corr_func.get_cf_by_cluster_names(
                atoms_sc, list(updated_cf.keys()))

            for key, value in brute_force.items():
                self.assertAlmostEqual(value, updated_cf[key])
        os.remove("sc4x4x.db")

    def test_double_swaps_ternary(self):
        if not has_ase_with_ce:  # Disable this test
            self.skipTest("ASE version has not cluster expansion")
            return

        db_name = "test_db_ternary.db"
        max_dia = get_max_cluster_dia_name()
        size_arg = {max_dia: 4.05}
        conc = Concentration(basis_elements=[["Al","Mg","Si"]])
        ceBulk = CEBulk(crystalstructure="fcc", a=4.05, size=[4,4,4],
                        concentration=conc, db_name=db_name, max_cluster_size=3, **size_arg)
        ceBulk.reconfigure_settings()
        corr_func = CorrFunction( ceBulk )
        cf = corr_func.get_cf( ceBulk.atoms )
        #prefixes = [name.rpartition("_")[0] for name in cf.keys()]
        #prefixes.remove("")
        eci = {name:1.0 for name in cf.keys()}
        atoms = ceBulk.atoms.copy()
        calc = CE(atoms, ceBulk, eci )
        atoms.set_calculator(calc)
        n_tests = 10

        # Insert 25 Mg atoms and 25 Si atoms
        n = 18
        for i in range(n):
            calc.calculate( atoms, ["energy"], [(i,"Al","Mg")])
            calc.calculate( atoms, ["energy"], [(i+n,"Al","Si")])
            updated_cf = calc.get_cf()
            brute_force = corr_func.get_cf_by_cluster_names( atoms, updated_cf.keys() )
            for key in updated_cf.keys():
                self.assertAlmostEqual( brute_force[key], updated_cf[key])

        # Swap atoms
        for i in range(n_tests):
            indx1 = np.random.randint(low=0,high=len(atoms))
            symb1 = atoms[indx1].symbol
            indx2 = indx1
            symb2 = symb1
            while( symb2 == symb1 ):
                indx2 = np.random.randint( low=0, high=len(ceBulk.atoms) )
                symb2 = atoms[indx2].symbol
            calc.calculate( atoms, ["energy"], [(indx1,symb1,symb2),(indx2,symb2,symb1)])
            update_cf = calc.get_cf()
            brute_force = corr_func.get_cf_by_cluster_names( atoms, update_cf.keys() )
            for key,value in brute_force.items():
                self.assertAlmostEqual( value, update_cf[key])

    def test_binary_spacegroup( self ):
        if ( not has_ase_with_ce ):
            self.skipTest("ASE version does not have CE")
            return
        bs, db_name = get_bulkspacegroup_binary()
        cf = CorrFunction( bs )
        cf_vals = cf.get_cf( bs.atoms )
        ecis = {name:1.0 for name in cf_vals.keys()}
        atoms = bs.atoms.copy()
        calc = CE(atoms, bs, ecis)

        for i in range(25):
            if ( bs.atoms[i].symbol == "Al" ):
                new_symb = "Mg"
                old_symb = "Al"
            else:
                new_symb = "Al"
                old_symb = "Mg"
            calc.calculate( atoms, ["energy"], [(i,old_symb,new_symb)] )
            updated_cf = calc.get_cf()
            brute_force = cf.get_cf_by_cluster_names( atoms, updated_cf.keys() )
            for key,value in brute_force.items():
                self.assertAlmostEqual( value, updated_cf[key] )

    def test_set_singlets( self ):
        if ( not has_ase_with_ce ):
            self.skipTest( "ASE version does not have CE" )
            return

        system_types = [["Al","Mg"],["Al","Mg","Si"],["Al","Mg","Si","Cu"]]

        db_name = "test_singlets.db"
        n_concs = 4
        no_throw = True
        msg = ""
        try:
            for basis_elems in system_types:
                a = 4.05
                mx_dia_name = get_max_cluster_dia_name()
                size_arg = {mx_dia_name:a}
                conc = Concentration(basis_elements=[basis_elems])
                ceBulk = CEBulk( crystalstructure="fcc", a=a, size=[5, 5, 5], concentration=conc, \
                db_name=db_name, max_cluster_size=2,**size_arg)
                ceBulk.reconfigure_settings()
                cf = CorrFunction(ceBulk)
                corrfuncs = cf.get_cf(ceBulk.atoms)
                eci = {name:1.0 for name in corrfuncs.keys()}
                atoms = ceBulk.atoms.copy()
                calc = CE(atoms, ceBulk, eci)
                for _ in range(n_concs):
                    conc = np.random.rand(len(basis_elems))*0.97
                    conc /= np.sum(conc)
                    conc_dict = {}
                    for i in range(len(basis_elems)):
                        conc_dict[basis_elems[i]] = conc[i]
                    calc.set_composition(conc_dict)
                    ref_cf = calc.get_cf()

                    singlets = {}
                    for key,value in ref_cf.items():
                        if ( key.startswith("c1") ):
                            singlets[key] = value
                    comp = calc.singlet2comp(singlets)
                    dict_comp = "Ref {}. Computed {}".format(conc_dict,comp)
                    for key in comp.keys():
                        self.assertAlmostEqual( comp[key], conc_dict[key], msg=dict_comp, places=1 )
                calc.set_singlets(singlets)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

    def test_sequence_of_swap_moves(self):
        if (not has_ase_with_ce):
            self.skipTest("ASE does not have CE")
            return
        atoms,ceBulk,eci = self.get_calc("fcc")
        calc = atoms.get_calculator()
        corr_func = CorrFunction(ceBulk)
        n_tests = 10

        # Insert 10 Mg atoms
        for i in range(n_tests):
            calc.calculate( atoms, ["energy"], [(i,"Al","Mg")] )

        # Swap Al and Mg atoms
        changes = []
        for i in range(n_tests):
            indx1 = i
            indx2 = len(atoms)-i-1
            symb1 = "Mg"
            symb2 = "Al"
            changes += [(indx1,symb1,symb2),(indx2,symb2,symb1)]

        calc.calculate( atoms, ["energy"], changes )
        updated_cf = calc.get_cf()
        brute_force = corr_func.get_cf_by_cluster_names( atoms, updated_cf.keys() )
        for key,value in brute_force.items():
            self.assertAlmostEqual( value, updated_cf[key] )

    def test_save_load_calculator(self):
        if not has_ase_with_ce:
            self.skipTest("ASE does not have CE")

        atoms, ceBulk, eci = self.get_calc("fcc")
        calc = atoms.get_calculator()
        calc_fname = "calculator.json"
        calc.save(calc_fname)
        calc2 = CE.load(calc_fname)
        os.remove(calc_fname)

        self.assertEqual(calc.BC.cluster_info, calc2.BC.cluster_info)
        self.assertEqual(calc.eci, calc2.eci)
        self.assertEqual(calc.get_cf(), calc2.get_cf())

        # Test to pickle a calculator
        import pickle
        pickled_string = pickle.dumps(calc)
        calc2 = pickle.loads(pickled_string)
        self.assertEqual(calc.BC.cluster_info, calc2.BC.cluster_info)
        self.assertEqual(calc.eci, calc2.eci)
        self.assertEqual(calc.get_cf(), calc2.get_cf())

    def test_load_from_JSON(self):
        if not has_ase_with_ce:
            self.skipTest("ASE does not have CE")
        _, ceBulk, eci = self.get_calc('fcc')

        outfile = 'ceBulkSettings.json'
        db_name = 'test_loadjson.db'
        ceBulk.save(outfile)
        atoms_full = get_atoms_with_ce_calc_JSON(
            outfile, eci=eci, size=[8, 9, 10], db_name=db_name)
        
        self.assertEqual(type(atoms_full.get_calculator()).__name__, 'CE')
        os.remove(db_name)
        os.remove(outfile)
        


if __name__ == "__main__":
    unittest.main(testRunner=TimeLoggingTestRunner)
