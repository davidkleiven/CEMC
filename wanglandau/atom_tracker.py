import numpy as np
class AtomTracker( object ):
    """
    Class that tracks which atoms are allowed to be swapped
    """
    def __init__( self, atoms, site_types=None, site_elements=None ):
        """
        Paramters
        ---------
        site_types: Array giving which site a given index is located
        site_elements: Array containing lists of all elements allowed on the different site types
        """
        self.atoms = atoms
        self.site_types = site_types
        self.site_elements = site_elements
        self.atom_position_track = None
        self.init_site_types()

    def init_site_types( self ):
        """
        Initialize the site types if not given
        """
        if ( self.site_types is None ):
            self.site_types = [0 for _ in range(len(self.atoms))]

    def init_position_tracker( self ):
        """
        Init the position tracker
        """
        number_of_site_types = np.max( self.site_types )+1
        self.atom_position_track = [{} for _ in range(number_of_site_types)]
        for atom in self.atoms:
            site_type = self.site_types[atom.index]
            if ( atom.symbol in self.atom_position_track[site_type].keys() ):
                self.atom_position_track[site_type][atom.symbol].append( atom.index )
            else:
                self.atom_position_track[site_type][atom.symbol] = [atom.index]

    def change_element( self, indx, new_element ):
        """
        Change the element on one positions
        """
        site_type = self.site_types[indx]
        if ( not new_element in self.site_elements ):
            raise RuntimeError( "The proposed change is not allowed" )

        
