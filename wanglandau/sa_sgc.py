import numpy as np
import ase.units
from ase.db import connect

class SimmualtedAnnealingSGC( object ):
    def __init__( self, atoms, chem_pot, db_name ):
        """
        Class for finding the ground state in the SGC ensemble
        """
        self.atoms = atoms
        self.chem_pot = chem_pot
        self.at_count = {key:0 for key in self.chem_pot.keys()}
        self.count_atoms()
        self.symbols = self.chem_pot.keys()
        self.energy = 0.0
        self.init_energy()
        self.kT = 100.0
        self.db_name = db_name

    def count_atoms( self ):
        """
        Counts the present number of atoms
        """
        for atom in self.atoms:
            if ( not atom.symbol in self.at_count.keys() ):
                raise ValueError( "A chemical potential for all species were not specified!" )
            self.at_count[atom.symbol] += 1

    def init_energy( self ):
        """
        Initialize with the energy of the first structure
        """
        self.energy = self.atoms.get_potential_energy()
        for key,value in self.chem_pot.iteritems():
            self.energy -= value*self.at_count[key]

    def _step( self ):
        """
        Perform one SA step
        """
        indx = np.random.randint(low=0,high=len(self.atoms))
        old_symb = self.atoms[indx].symbol

        new_symb = old_symb
        while ( new_symb == old_symb ):
            indx2 = np.random.randint( low=0,high=len(self.symbols) )
            new_symb = self.symbols[indx2]

        # Swap symbols
        self.atoms[indx].symbol = new_symb
        new_energy = self.atoms.get_potential_energy()
        dE = (new_energy-self.energy) - self.chem_pot[new_symb] + self.chem_pot[old_symb]
        rand_num = np.random.rand()
        if ( rand_num < np.exp(-dE/(self.kT)) ):
            self.energy = new_energy
            self.at_count[new_symb] += 1
            self.at_count[old_symb] -= 1
        else:
            self.atoms[indx].symbol = old_symb

    def run( self, ntemps=5, n_steps=10000, Tmin=1, Tmax=10000 ):
        """
        Perform SA to estimate the ground state energy and structure
        """
        kTs = ase.units.kB*np.logspace(np.log10(Tmin),np.log10(Tmax),ntemps)[::-1]
        for step in range(ntemps):
            self.kT = kTs[step]
            print ("Current T: %.2E"%(self.kT/ase.units.kB) )
            for _ in range(n_steps):
                self._step()

        db = connect( self.db_name )
        energy = self.atoms.get_potential_energy()
        elm = [key for key,value in self.chem_pot.iteritems()]
        pot = [value for key,value in self.chem_pot.iteritems()]
        uid = db.write( self.atoms, Tmin=Tmin, Tmax=Tmax, data={
        "elements":elm,
        "chemical_potentials":pot
        } )
