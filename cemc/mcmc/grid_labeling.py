import numpy as np

class GridLabeling(object):
    def __init__(self, atoms=None, bin_length=3.0, track_elements=[]):
        self.atoms = atoms
        self.bin_length = bin_length
        self.track_elements = track_elements
        
        lengths = self.atoms.get_cell_lengths_and_angles()[:3]
        size = tuple(lengths/self.bin_length)
        self.bin_shape = self.atoms.get_cell()
        self.bin_shape[:, 0] /= size[0]
        self.bin_shape[:, 1] /= size[1]
        self.bin_shape[:, 2] /= size[2]
        self.inv_bin_shape = np.linalg.inv(self.bin_shape)
        self.bins = np.zeros(size, dtype=np.uint8)
        self.pos = self.atoms.get_positions()

    def get_bin(self, atom_indx):
        """Return the bin given an atom index."""
        x = self.pos[atom_indx, :]
        n = self.inv_bin_shape.dot(x)
        return n.astype(np.int32)

    def populate_grid(self):
        """Loop over atoms object and populate the grid.
           
           The algrotihm will count how many of each element
           there is in each bin
        """
        self.bins[:, :, :] = 0
        for atom in self.atoms:
            if atom.symbol in self.track_elements:
                n = self.get_bin(atom.index)
                self.bins[n[0], n[1], n[2]] += 1
