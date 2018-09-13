import numpy as np
from random import choice


class SwapMoveIndexTracker(object):
    def __init__(self):
        self.symbols = []
        self.tracker = {}
        self.index_loc = None

    def _symbols_from_atoms(self, atoms):
        symbs = [atom.symbol for atom in atoms]
        return list(set(symbs))

    def _count_symbs(self):
        symb_count = {symb: 0 for symb in self.symbols}
        for atom in self.atoms:
            symb_count[atom.symbol] += 1
        return symb_count

    def init_tracker(self, atoms):
        """Initialize the tracker with the numbers."""
        self.symbols = self._symbols_from_atoms(atoms)

        # Track indices of all symbols
        self.tracker = {symb: [] for symb in self.symbols}

        # Track the location in self.tracker of each index
        self.index_loc = np.zeros(len(atoms), dtype=int)
        for atom in atoms:
            self.tracker[atom.symbol].append(atom.index)
            self.index_loc[atom.index] = len(self.tracker[atom.symbol])-1

    def update_swap_move(self, system_changes):
        """Update the atoms tracker."""
        indx1 = system_changes[0][0]
        indx2 = system_changes[1][0]
        symb1 = system_changes[0][1]
        symb2 = system_changes[0][2]

        # Find the locations of the indices
        loc1 = self.index_loc[indx1]
        loc2 = self.index_loc[indx2]

        # Update the tracker and the locations
        self.tracker[symb1][loc1] = indx2
        self.index_loc[indx2] = loc1

        self.tracker[symb2][loc2] = indx1
        self.index_loc[indx1] = loc2

    def get_random_indx_of_symbol(self, symbol):
        return choice(self.tracker[symbol])
