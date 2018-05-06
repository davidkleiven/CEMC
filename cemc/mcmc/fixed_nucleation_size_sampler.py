from montecarlo import Montecarlo
from mc_observers import NetworkObserver
import numpy as np

class FixedNucleusMC( Montecarlo ):
    def __init__( self, atoms, T, **kwargs ):
        self.size = kwargs.pop("size")
        self.network_element = kwargs.pop("network_element")
        self.network_name = kwargs.pop("network_name")

        super(FixedNucleusMC,self).__init__( atoms, T, **kwargs )
        self.network = NetworkObserver( calc=self.atoms._calc, cluster_name=self.network_name, element=self.network_element )
        self.bc = self.atoms._calc.BC
        self.network_clust_indx = None
        self.find_cluster_indx()
        print (self.network_clust_indx)

    def find_cluster_indx( self ):
        """
        Find the cluster indices corresponding to the current network name
        """
        indx = self.bc.cluster_names[0][2].index(self.network_name)
        self.network_clust_indx = self.bc.cluster_indx[0][2][indx]

    def get_trial_move(self):
        """
        Generate a trial move
        """
        Na = len(self.atoms_indx[self.network_element])


        symb_b = self.network_element
        max_rand = len(self.network_clust_indx)
        while ( symb_b == self.network_element ):
            self.selected_a = np.random.randint(low=0,high=Na)
            self.rand_a = self.atoms_indx[self.network_element][self.selected_a]

            temp_rand_num_b = np.random.randint(low=0,high=Na)
            temp_b = self.atoms_indx[self.network_element][temp_rand_num_b]
            indx = np.random.randint(low=0,high=max_rand)
            clust_indx = self.network_clust_indx[indx][0]
            self.rand_b = self.bc.trans_matrix[temp_b,clust_indx]
            symb_b = self.atoms[self.rand_b].symbol

        #self.selected_b = self.atoms_indx[symb_b].index(self.rand_b)
        self.selected_b = 0 # Breaks the cluster tracker but it is not needed here
        symb_a = self.atoms[self.rand_a].symbol
        symb_b = self.atoms[self.rand_b].symbol
        system_changes = [(self.rand_a,symb_a,symb_b),(self.rand_b,symb_b,symb_a)]
        return system_changes

    def accept( self, system_changes ):
        """
        Accept trial move
        """
        move_accepted = Montecarlo.accept( self, system_changes )
        self.network.reset()
        self.network(system_changes)
        stat = self.network.get_statistics()
        self.network.reset()
        if ( stat["max_size"] != self.size ):
            return False
        return move_accepted

    def get_atoms(self, atoms=None, prohib_elem=[]):
        """
        Return the atoms object with the clusters highlighted
        """
        #atoms = self.network.get_atoms_with_largest_cluster()
        if ( atoms is None ):
            atoms = self.atoms
        else:
            symbs = [atom.symbol for atom in atoms]
            self.atoms._calc.set_symbols(symbs)
            atoms = self.atoms

        self.network.reset()
        self.network(None)
        indices = self.network.get_indices_of_largest_cluster()

        ref_atom = indices[0]
        pos = atoms.get_positions()
        mic_dists = atoms.get_distances( ref_atom, indices, mic=True, vector=True )
        com = pos[ref_atom,:] + np.mean(mic_dists,axis=0)
        cell = atoms.get_cell()
        center = 0.5*(cell[0,:]+cell[1,:]+cell[2,:])
        atoms.translate( center-com )
        atoms.wrap()
        return atoms, atoms[indices]


    def run( self, nsteps=100000, init_cluster=True ):
        """
        Run Monte Carlo for fixed nucleus size
        """
        if ( init_cluster ):
            self.network.grow_cluster(self.size)
            self.build_atoms_list()
            self.atoms._calc.clear_history()
        step = 0
        while( step < nsteps ):
            step += 1
            self._mc_step()
        print ("Acceptance rate: {}".format(float(self.num_accepted)/self.current_step))
