import numpy as np
import ase.units
from ase.db import connect
from ce_calculator import CE
import time
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plt

class SimmualtedAnnealingSGC( object ):
    def __init__( self, atoms, db_name, chem_pot=None ):
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
        self.status_every_sec = 60
        self.visiting_statistics = np.zeros(len(self.atoms))
        self.comps = {key:[] for key in self.at_count.keys()}
        self.iter = 1

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
        system_changes = [(indx,old_symb,new_symb)]
        #new_energy = self.atoms.get_potential_energy( system_changes=system_changes )
        new_energy = self.atoms._calc.calculate( self.atoms, ["energy"], system_changes )
        dE = (new_energy-self.energy) - self.chem_pot[new_symb] + self.chem_pot[old_symb]
        self.visiting_statistics[indx] += 1
        if ( dE < 0.0 ):
            accepted = True
        else:
            accepted = np.random.rand() < np.exp(-dE/(self.kT))

        if ( accepted ):
            self.energy = new_energy
            self.at_count[new_symb] += 1
            self.at_count[old_symb] -= 1

            if ( isinstance(self.atoms._calc,CE) ):
                self.atoms._calc.clear_history()
        else:
            self.atoms[indx].symbol = old_symb
            if ( isinstance( self.atoms._calc, CE) ):
                self.atoms._calc.undo_changes()

        N = len(self.atoms)
        self.comps[new_symb].append(self.at_count[new_symb]/float(N))
        self.comps[old_symb].append(self.at_count[old_symb]/float(N))
        self.iter += 1

    def run( self, ntemps=5, n_steps=10000, Tmin=1, Tmax=10000 ):
        """
        Perform SA to estimate the ground state energy and structure
        """
        kTs = ase.units.kB*np.logspace(np.log10(Tmin),np.log10(Tmax),ntemps)[::-1]
        for step in range(ntemps):
            self.kT = kTs[step]
            print ("Current T: %.2E"%(self.kT/ase.units.kB) )
            last_step = 0
            start = time.time()
            for curr in range(n_steps):
                if ( time.time()-start > self.status_every_sec ):
                    print ("%d of %d steps. %.2f ms per step"%(curr,n_steps,self.status_every_sec*1000.0/float(curr-last_step)) )
                    last_step = curr
                    start = time.time()
                self._step()
        print (self.at_count)
        db = connect( self.db_name )
        energy = self.atoms.get_potential_energy()
        elm = [key for key,value in self.chem_pot.iteritems()]
        pot = [value for key,value in self.chem_pot.iteritems()]
        uid = db.write( self.atoms, Tmin=Tmin, Tmax=Tmax, data={
        "elements":elm,
        "chemical_potentials":pot
        } )

    def show_visit_stat( self ):
        """
        Creates a histogram over the number of times each atom have been
        visited
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.visiting_statistics, ls="steps" )
        ax.set_xlabel( "Atom index" )
        ax.set_ylabel( "Number of times visited" )
        return fig

    def show_compositions( self ):
        """
        Plots how the composition changes during the simmulated annealing
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for key in self.comps.keys():
            cumulative = np.cumsum(self.comps[key])
            cumulative /= (np.arange(len(cumulative))+1.0)
            ax.plot( self.comps[key], label=key )
        ax.set_xlabel( "MC step" )
        ax.set_ylabel( "Concentration" )
        ax.legend( loc="best", frameon=False )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return fig
