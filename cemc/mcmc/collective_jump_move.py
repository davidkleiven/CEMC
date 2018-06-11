import numpy as np
from ase.spacegroup import get_spacegroup


class CollectiveJumpMove(object):
    def __init__( self, mc_cell=None, a=4.05 ):
        """
        Extracts the indices of that atoms in each column that can
        perform a jump in the z-direction

        This is targeted at FCC cells having the possibility to perform
        the jump described

        Marioara, C. D., et al. "Atomic model for GP-zones in a 6082 Al-Mg-Si system." Acta materialia 49.2 (2001): 321-328.
        """
        if (mc_cell is None):
            raise TypeError("No Monte Carlo Cell is given")
        self.mc_cell = mc_cell
        self._check_atoms()
        self.columns = {}
        self.column_state = {}
        self.col_ids = {}
        self.a = a
        self._extract_indices_in_column()
        self.columns_are_valid()

    def _check_atoms(self):
        """
        Check that the atoms can perform a jump move
        """
        sp = get_spacegroup(self.mc_cell)
        if ( sp.no != 221 ):
            raise ValueError("Only spacegroup 221 can perform this move")

    def _can_perform_jump(self, pos):
        """
        Returns True if this atom can perform a jump move
        """
        a = self.a
        nx = int(np.round(pos[0]/a))
        x = (pos[0]-0.5*a)/a
        ny = int(np.round(pos[1]/a))
        y = (pos[1]-0.5*a)/a
        eps = 1E-6
        return np.abs(nx-x) < eps and np.abs(ny-y) < eps

    def _get_column_key(self, pos):
        """
        Returns the key to the corresponding key dictionary
        """
        a = self.a
        nx = int(np.round((pos[0]-0.5*a)/a))
        ny = int(np.round((pos[1]-0.5*a)/a))
        return "{}_{}_{}".format(nx,ny,"z")

    def _extract_indices_in_column(self):
        """
        Extract indices in column
        """
        # The atoms that can perform jump moves are located at (n*a*(1+a/2),n*a(1+1/2),z)
        pos = self.mc_cell.get_positions()
        zpos = {}
        for i in range(pos.shape[0]):
            if ( self._can_perform_jump(pos[i,:])):
                col_key = self._get_column_key(pos[i,:])
                if ( col_key not in self.columns.keys() ):
                    self.columns[col_key] = []
                    zpos[col_key] = []
                self.columns[col_key].append(i)
                self.col_ids[i] = col_key
                zpos[col_key].append(col_key)

        for key in self.columns.keys():
            srt_indx = np.argsort(zpos[key])
            self.columns[key] = [self.columns[key][indx] for indx in srt_indx]
        self.column_state = {key:"down" for key in self.columns.keys()}

    def columns_are_valid(self):
        """
        Verify that all columns are valid. They need to have a 50 percent vacancy content
        """
        for (key,value) in self.columns.items():
            n_vac = 0
            for indx in value:
                if ( self.mc_cell[indx].symbol == "X" ):
                    n_vac += 1
            if ( n_vac != int(0.5*len(value))):
                msg = "The column is not valid. They need to have 50 percent vacancies\n"
                msg += "N_vac in column {}: {} of {}".format(key,n_vac,len(value))
                raise ValueError(msg)

            # Check that every other symbol is X
            first = [self.mc_cell[indx].symbol for indx in value[::2]]
            second = [self.mc_cell[indx].symbol for indx in value[1::2]]
            all_symbs = [self.mc_cell[indx].symbol for indx in value]
            first_has_vac = (first[0]=="X")
            second_has_vac = (second[0]=="X")
            if ( first_has_vac ):
                vacancies = first
                elements = second
            elif (second_has_vac):
                vacancies=second
                elements=first
            else:
                raise ValueError("Vacancies are not ordered. {}".format(all_symbs))
            for i in range(len(first)):
                if (vacancies[i] != "X" or elements[i]=="X"):
                    raise ValueError("Vacancies are not orderered. {}".format(all_symbs))

    def system_changes_random_jump_move(self):
        """
        Select a random jumb move
        """
        keys = self.columns.keys()
        key = keys[np.random.randint(low=0,high=len(keys))]
        syst_changes = []
        indices = self.columns[key]
        for num in range(len(indices)):
            if ( num%2 == 0 ):
                # Swap with the atom above
                change = (indx,self.mc_cell[indices[num]],self.mc_cell[indices[num+1]])
            else:
                # Swap with the one below
                change = (indx,self.mc_cell[indices[num]],self.mc_cell[indices[num-1]])
            syst_changes.append(change)
        return syst_changes

    def view_columns(self):
        """
        View all columns
        """
        from ase.gui.gui import GUI
        from ase.gui.images import Images
        all_columns = []
        for (key,indices) in self.columns.items():
            col = self.mc_cell[indices]
            col.info = {"name":key}
            all_columns.append(col)
        images = Images()
        images.initialize(all_columns)
        gui = GUI(images)
        gui.show_name = True
        gui.run()
