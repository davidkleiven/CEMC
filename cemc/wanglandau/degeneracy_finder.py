from ase.spacegroup.spacegroup import SpaceGroup
from cemc.wanglandau.wltools import count_elements
import numpy as np
from scipy.spatial import cKDTree as KDTree

class DegeneracyFinder( object ):
    """
    This class finds all UNIQUE symmetrically equivalent states
    """
    def __init__( self, atoms, spgroup ):
        self.atoms = atoms
        self.sg = SpaceGroup(spgroup)
        self.unique_gs = [self.atoms]
        self.pos_tree = [KDTree(self.atoms.get_positions())]

    def get_degeneracy( self ):
        """
        Returns the number of degenerate structures
        """
        atoms_count = count_elements( self.atoms )
        if ( len(atoms_count.keys()) == 1 ):
            return 1
        elif ( len(atoms_count.keys()) == 2 ):
            for key,value in atoms_count:
                if ( value == 1 ):
                    return len(atoms)
        else:
            # Find the number of generate structures by brute force
            # TODO: Is there a general way of quickly returning this if the
            # spacegroup and one structure is known?
            self.find_unique_gs()
            return len(self.unique_gs)

    def already_found( self, atoms ):
        """
        Returns True if this structure already has been registered
        """
        pos = atoms.get_positions()
        for indx,tree in enumerate(self.pos_tree):
            dist, indices = tree.query(pos)
            for i in range(len(indices)):
                if ( atoms[i].symbol != self.unique_gs[i][indices[i]] ):
                    return False
        return True

    def register( self, atoms ):
        """
        Registers a new structure
        """
        self.unique_gs.append( atoms )
        self.pos_tree.append( KDTree(atoms.get_positions()) )

    def find_unique_gs( self ):
        """
        Finds all unique ground state structures and puts them into a list of unique_gs
        """
        cell = self.atoms.get_cell().T
        inv_cell = np.linalg.inv(cell)
        ops = self.sg.get_rotations()
        atom_cp = self.atoms.copy()
        for i in range( len(atom_cp) ):
            pos = self.atoms.get_positions()
            atom_cp.set_positions( pos-pos[i,:] )
            atom_cp.wrap()
            for op in ops:
                trans_op = cell.dot(op).dot(inv_cell)
                pos = atom_cp.get_positions()
                new_pos = trans_op.dot(pos.T).T
                atom_cp.set_positions( new_pos )
                atom_cp.wrap()
                if ( not self.already_found(atom_cp) ):
                    self.register( atom_cp.copy() )
