from ase.calculators.calculator import Calculator
from ase.ce.corrFunc import CorrFunction
from ase.ce.settings import BulkCrystal
from ase.build import bulk
import unittest
from itertools import product, combinations
import os
import numpy as np
import copy
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plt
from ase.visualize import view
try:
    from cemc.ce_updater import ce_updater as ce_updater
    use_cpp = True
except Exception as exc:
    use_cpp = False
    print (str(exc))
    print ("Could not find C++ version, falling back to Python version")

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
            self.cf = self.corrFunc.get_cf_by_cluster_names(self.atoms,eci.keys())
        else:
            self.cf = initial_cf

        # Make sure that the database information fits
        if ( len(BC.atoms) != BC.trans_matrix.shape[0] ):
            raise ValueError( "The number of atoms and the dimension of the translation matrix is inconsistent. Try reconf_db=True in bulk crystal" )
        self.old_cfs = []
        self.old_atoms = self.atoms.copy()
        self.eci = eci
        self.changes = []
        self.ctype = {}
        self.create_ctype_lookup()
        self.convert_cluster_indx_to_list()
        self.permutations = {}
        self.create_permutations()
        self.BC.trans_matrix = np.array(self.BC.trans_matrix).astype(np.int32)
        self.updater = None
        if ( use_cpp ):
            self.updater = ce_updater.CEUpdater()
            self.updater.init( self.BC, self.cf, self.eci, self.permutations )

            if ( not self.updater.ok() ):
                raise RuntimeError( "Could not initialize C++ CE updater" )

        if ( use_cpp ):
            self.clear_history = self.updater.clear_history
            self.undo_changes = self.updater.undo_changes
            self.update_cf = self.updater.update_cf
        else:
            self.clear_history = self.clear_history_pure_python
            self.undo_changes = self.undo_changes_pure_python
            self.update_cf = self.update_cf_pure_python

    def convert_cluster_indx_to_list( self ):
        """
        Converts potentials arrays to lists
        """
        for i in range(len(self.BC.cluster_indx)):
            if ( self.BC.cluster_indx[i] is None ):
                continue
            for j in range(len(self.BC.cluster_indx[i])):
                if ( self.BC.cluster_indx[i][j] is None ):
                    continue
                for k in range(len(self.BC.cluster_indx[i][j])):
                    if ( isinstance(self.BC.cluster_indx[i][j][k],np.ndarray) ):
                        self.BC.cluster_indx[i][j][k] = self.BC.cluster_indx[i][j][k].tolist()
                    else:
                        self.BC.cluster_indx[i][j][k] = list(self.BC.cluster_indx[i][j][k])

                if ( isinstance(self.BC.cluster_indx[i][j],np.ndarray) ):
                    self.BC.cluster_indx[i][j] = self.BC.cluster_indx[i][j].tolist()
                else:
                    self.BC.cluster_indx[i][j] = list(self.BC.cluster_indx[i][j])

            if ( isinstance(self.BC.cluster_indx[i],np.ndarray) ):
                self.BC.cluster_indx[i] = self.BC.cluster_indx[i].tolist()
            else:
                self.BC.cluster_indx[i] = list(self.BC.cluster_indx[i])

    def create_permutations( self ):
        """
        Creates a list of permutations of basis functions that should be passed
        to the C++ module
        """
        bf_list = list(range(len(self.BC.basis_functions)))
        for num in range(2,len(self.BC.cluster_names)):
            perm = list(product(bf_list, repeat=num))
            #perm = list(combinations(bf_list, repeat=num))
            self.permutations[num] = perm


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

    def update_cf_pure_python( self, single_change ):
        """
        Changing one element and update the correlation functions
        """
        indx = single_change[0]
        old_symb = single_change[1]
        new_symb = single_change[2]
        self.old_cfs.append( copy.deepcopy(self.cf) )
        if ( old_symb == new_symb ):
            return self.cf
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

    def undo_changes_pure_python( self ):
        """
        This function undo all changes stored in all symbols starting from the
        last one
        """
        for i in range(len(self.changes),0,-1):
            entry = self.changes[i-1]
            self.atoms[entry[0]].symbol = entry[1]
            self.cf = self.old_cfs[i-1]
        self.clear_history()

    def clear_history_pure_python( self ):
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
        if ( use_cpp ):
            energy = self.updater.calculate(system_changes)
            self.cf = self.updater.get_cf()
        else:
            self.changes += system_changes
            for entry in system_changes:
                self.update_cf( entry )
            energy = self.get_energy()
        self.results["energy"] = energy
        return self.results["energy"]

    def get_cf( self ):
        """
        Returns the correlation functions
        """
        if ( self.updater is None ):
            return self.cf
        else:
            return self.updater.get_cf()

    def update_ecis( self, new_ecis ):
        """
        Updates the ecis
        """
        self.eci = new_ecis
        if ( not self.updater is None ):
            self.updater.set_ecis(self.eci)

    def get_singlets( self, array ):
        if ( self.updater is None ):
            indx = 0
            for key,value in self.cf.iteritems():
                if ( key.startswith("c1") ):
                    singlets[indx] = value
                    indx += 1
            return array
        else:
            self.updater.get_singlets( array )
            return array

    def set_composition( self, comp ):
        """
        Change composition of an object.
        """
        # Verify that the sum of the compositions is one
        tot_conc = 0.0
        for key,conc in comp.iteritems():
            tot_conc += conc

        if ( tot_conc != 1.0 ):
            raise ValueError( "The specified concentration does not sum to 1!" )
        # Change all atoms to the first one
        init_elm = comp.keys()[0]
        for i in range( len(self.atoms) ):
            self.update_cf( (i,self.atoms[i].symbol,init_elm) ) # Set all atoms to init element
        start = 0
        for elm,conc in comp.iteritems():
            if ( elm == init_elm ):
                continue
            n_at = int( conc*len(self.atoms) )
            for i in range(start,start+n_at):
                self.update_cf( (i,init_elm,elm) )
            start += n_at
        self.clear_history()

    def set_symbols( self, symbs ):
        """
        Change the symbols of the entire atoms object
        """
        if ( len(symbs) != len(self.atoms ) ):
            raise ValueError( "Length of the symbols array has to match the length of the atoms object.!" )
        for i,symb in enumerate(symbs):
            self.update_cf( (i,self.atoms[i].symbol,symb) )
        self.clear_history()
################################################################################
##                           UNIT TESTS                                       ##
################################################################################
