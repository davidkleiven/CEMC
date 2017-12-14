import sys
sys.path.insert( 1, "/home/davidkl/Documents/aseJin")
from ase.calculators.calculator import Calculator
from ase.ce.corrFunc import CorrFunction
from ase.ce.settings import BulkCrystal
from ase.build import bulk
import unittest
from itertools import product
import os
import numpy as np
import copy



class CE( Calculator ):
    """
    Class for updating the CE when symbols change
    """

    implemented_properties = ["energy"]
    def __init__( self, BC, eci, initial_cf=None ):
        Calculator.__init__( self )
        self.BC = BC
        self.corrFunc = CorrFunction(self.BC)
        self.atoms = self.BC.atoms
        if ( initial_cf is None ):
            self.cf = self.corrFunc.get_cf(self.atoms)
        else:
            self.cf = initial_cf
        self.old_cfs = []
        self.old_atoms = self.atoms.copy()
        self.eci = eci
        self.changes = []
        self.ctype = {}
        self.create_ctype_lookup()

    def get_energy( self ):
        """
        Returns the energy of the system
        """
        energy = 0.0
        for key,value in self.eci.iteritems():
            energy += value*self.cf[key]
        return energy*len(self.atoms)

    def create_ctype_lookup( self ):
        """
        Creates a lookup table for cluster types based on the prefix
        """
        for n in range(2,len(self.BC.cluster_names)):
            for ctype in range(len(self.BC.cluster_names[n])):
                name = self.BC.cluster_names[n][ctype]
                prefix = name#name.rpartition('_')[0]
                self.ctype[prefix] = (n,ctype)

    def update_cf( self, indx, old_symb, new_symb ):
        """
        Changing one element and update the correlation functions
        """
        if ( old_symb == new_symb ):
            return self.cf
        self.old_cfs.append( copy.deepcopy(self.cf) )
        natoms = len(self.atoms)
        bf_list = list(range(len(self.BC.basis_functions)))

        self.atoms[indx].symbol = new_symb
        bf = self.BC.basis_functions
        for name in self.eci.keys():
            if ( name == "c0" ):
                continue
            elif ( name.startswith("c1") ):
                dec = int(name[-1]) - 1
                self.cf[name] += (bf[dec][new_symb]-bf[dec][old_symb])/natoms
                continue
            prefix = name.rpartition('_')[0]
            dec = int(name.rpartition('_')[-1]) - 1

            res = self.ctype[prefix]
            num = res[0]
            ctype = res[1]
            #for n in range(2, len(self.BC.cluster_names)):
            #    try:
            #        ctype = self.BC.cluster_names[n].index(prefix)
            #        num = n
            #        break
            #    except ValueError:
            #        continue
            perm = list(product(bf_list, repeat=num))
            count = len(self.BC.cluster_indx[num][ctype])*natoms
            sp = self.spin_product_one_atom( indx, self.BC.cluster_indx[num][ctype], perm[dec] )
            sp /= count
            bf_indx = perm[dec][0]
            self.cf[name] += num*( bf[bf_indx][new_symb] - bf[bf_indx][old_symb] )*sp
        return self.cf

    def spin_product_one_atom( self, ref_indx, indx_list, dec ):
        """
        Spin product for a single atom
        """
        num_indx = len(indx_list)
        bf = self.BC.basis_functions
        sp = 0.0
        for i in range(num_indx):
            sp_temp = 1.0
            for j, indx in enumerate(indx_list[i][:]):
                trans_indx = self.corrFunc.trans_matrix[ref_indx, indx]
                sp_temp *= bf[dec[j+1]][self.atoms[trans_indx].symbol]
            sp += sp_temp
        return sp

    def undo_changes( self ):
        """
        This function undo all changes stored in all symbols starting from the
        last one
        """
        for i in range(len(self.changes),0,-1):
            entry = self.changes[i-1]
            self.atoms[entry[0]].symbol = entry[1]
            self.cf = self.old_cfs[i-1]
        self.clear_history()

    def clear_history( self ):
        """
        Clears the history of the calculator
        """
        self.changes = []
        self.old_cfs = []

    def calculate( self, atoms, properties, system_changes ):
        """
        Calculates the energy. The system_changes is assumed to be a list
        of tuples of the form (indx,old_symb,new_symb)
        """
        for entry in system_changes:
            self.changes.append( (entry[0],entry[1],entry[2]) )
            self.update_cf( entry[0], entry[1], entry[2] )
        self.results["energy"] = self.get_energy()
        return self.results["energy"]

################################################################################
##                           UNIT TESTS                                       ##
################################################################################
class TestCE( unittest.TestCase ):
    def test_update( self ):
        db_name = "test_db.db"
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        eci = {}
        ceBulk = BulkCrystal( "fcc", 4.05, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
        calc = CE( ceBulk, eci )
        calc.eci = {key:0.0 for key in calc.cf}
        eci = calc.eci
        n_tests = 10
        for i in range(n_tests):
            updated_cf = calc.update_cf( i+1, "Al", "Mg" )
            calc = CE( ceBulk, eci ) # Now the calculator is initialized with the new atoms object
            for key,value in updated_cf.iteritems():
                self.assertAlmostEqual( value, calc.cf[key] )

    def test_random_swaps( self ):
        db_name = "test_db.db"
        conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }
        ceBulk = BulkCrystal( "fcc", 4.05, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
        eci = {}
        calc = CE( ceBulk, eci )
        calc.eci = {key:0.0 for key in calc.cf}
        eci = calc.eci
        n_tests = 100

        for i in range(n_tests):
            print ("%d of %d random tests"%(i+1,n_tests))
            indx = np.random.randint(low=0,high=len(ceBulk.atoms))
            old_symb = ceBulk.atoms[indx].symbol
            if ( old_symb == "Al" ):
                new_symb = "Mg"
            else:
                new_symb = "Al"
            updated_cf = calc.update_cf( indx, old_symb, new_symb )
            brute_force_calc = CE( ceBulk, eci )
            for key,value in updated_cf.iteritems():
                self.assertAlmostEqual( value, brute_force_calc.cf[key] )

if __name__ == "__main__":
    unittest.main()
