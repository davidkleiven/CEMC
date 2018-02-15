from ase.spacegroup.spacegroup import get_spacegroup
import numpy as np
class GroundStateDegeneracy( object ):
    def __init__( self, atoms ):
        self.atoms = atoms.copy()
        self.spgroup = 1
        self.find_space_group()

    def find_space_group( self ):
        """
        Determines the space group
        """
        atom_cp = self.atoms.copy()
        for atom in atom_cp:
            atom.symbol = "Al" # Set all symbols to the same
        self.spgroup = get_spacegroup(atom_cp)

    def find_degenerate_structures( self ):
        """
        Find all degenerate structures
        """
        atom_cp = self.atoms.copy()
        ops = self.get_rotations()
        cell = self.atoms.get_cell().T
        invcell = np.linalg.inv(cell)

        for i in range(0,len(atom_cp)):
            orig_pos = self.atoms.get_positions()
            atom_cp.set_positions( orig_pos-orig_pos[i,:] )
            atom_cp.wrap()
            for op in ops:
