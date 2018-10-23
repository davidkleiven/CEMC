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
        self.clusters = np.zeros_like(self.bins)

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
        from cemc_cpp_code import hoshen_kopelman
        self.bins[:, :, :] = 0
        for atom in self.atoms:
            if atom.symbol in self.track_elements:
                n = self.get_bin(atom.index)
                self.bins[n[0], n[1], n[2]] += 1

        # Run the Hoshen-Kopelman algorithm to label the 
        # bins into clusters
        self.clusters = hoshen_kopelman(self.bins)

    @property
    def num_clusters(self):
        return np.max(self.clusters)

    @property
    def cluster_sizes(self):
        return np.bincount(self.clusters)

    def get_bin_changes(self, system_changes):
        """Return the changes to the bins produced by this move."""
        bin_changes = {}
        for change in system_changes:
            new_bin = tuple(self.get_bin(change[0]))
            if change[1] not in self.track_elements and change[2] in self.track_elements:
                bin_changes[new_bin] = bin_changes.get(new_bin, 0) + 1
            elif change[1] in self.track_elements and change[2] not in self.track_elements:
                bin_changes[new_bin] = bin_changes.get(new_bin, 0) - 1
        return bin_changes

    def move_require_cluster_update(self, system_changes):
        """Return True if the move require updating the clusters."""
        bin_changes = self.get_bin_changes(system_changes)
        
        for k, v in bin_changes.items():
            if self.bins[k] == 0 and v > 0:
                # An empty bin becomes occupied
                return True
            elif v < 0 and abs(v) >= self.bins[k]:
                # An occupied bin becomes unoccupied
                return True

        # No new occupied bins are created by this move
        # and no occupied bins become unoccupied
        return False

