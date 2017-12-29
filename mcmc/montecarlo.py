# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division

import numpy as np
import ase.units as units
from wanglandau import ce_calculator
import time
import logging
#from ase.io.trajectory import Trajectory



class Montecarlo:
    """ Class for performing MonteCarlo sampling for atoms

    """

    def __init__(self,atoms,temp,indeces = None):
        """ Initiliaze Monte Carlo simulations object

        Arguments:
        atoms : ASE atoms object
        temp  : Temperature of Monte Carlo simulation in Kelvin
        indeves; List of atoms involved Monte Carlo swaps. default is all atoms.

        """
        self.atoms = atoms
        self.T = temp
        if indeces == None:
            self.indeces = range(len(self.atoms))
        else:
            self.indeces = indeces

        self.observers = [] # List of observers that will be called every n-th step
                            # similar to the ones used in the optimization routines

        self.current_step = 0
        self.status_every_sec = 60
        self.atoms_indx = {}
        self.symbols = []
        self.build_atoms_list()
        self.current_energy = 1E100

    def build_atoms_list( self ):
        """
        Creates a dictionary of the indices of each atom which is used to
        make sure that two equal atoms cannot be swapped
        """
        for atom in self.atoms:
            if ( not atom.symbol in self.atoms_indx.keys() ):
                self.atoms_indx[atom.symbol] = [atom.index]
            else:
                self.atoms_indx[atom.symbol].append(atom.index)
        self.symbols = self.atoms_indx.keys()

    def attach( self, obs, interval=1 ):
        """
        Attach observers that is called on each MC step
        and receives information of which atoms get swapped
        """
        if ( callable(obs) ):
            self.observers.append( (interval,obs) )


    def runMC(self,steps = 10, verbose = False ):
        """ Run Monte Carlo simulation

        Arguments:
        steps : Number of steps in the MC simulation

        """

        # Atoms object should have attached calculator
        # Add check that this is show
        self.current_energy = 1E8
        self._mc_step()
        #self.current_energy = self.atoms.get_potential_energy() # Get starting energy

        totalenergies = []
        totalenergies.append(self.current_energy)
        start = time.time()
        prev = 0
        step = 0
        while( step < steps ):
            step += 1
            en, accept = self._mc_step( verbose=verbose )
            #totalenergies.append(en)

            if ( time.time()-start > self.status_every_sec ):
                print ("%d of %d steps. %.2f ms per step"%(step,steps,1000.0*self.status_every_sec/float(step-prev)))
                prev = step
                start = time.time()
        return totalenergies


    def _mc_step(self, verbose = False ):
        """
        Make one Monte Carlo step by swithing two atoms
        """
        number_of_atoms = len(self.atoms)

        rand_a = self.indeces[np.random.randint(0,len(self.indeces))]
        rand_b = self.indeces[np.random.randint(0,len(self.indeces))]
        symb_a = self.symbols[np.random.randint(0,len(self.symbols))]
        symb_b = symb_a
        while ( symb_b == symb_a ):
            symb_b = self.symbols[np.random.randint(0,len(self.symbols))]

        Na = len(self.atoms_indx[symb_a])
        Nb = len(self.atoms_indx[symb_b])
        selected_a = np.random.randint(0,Na)
        selected_b = np.random.randint(0,Nb)
        rand_a = self.atoms_indx[symb_a][selected_a]
        rand_b = self.atoms_indx[symb_b][selected_b]

        # TODO: The MC calculator should be able to have constraints on which
        # moves are allowed. CE requires this some elements are only allowed to
        # occupy some sites
        symb_a = self.atoms[rand_a].symbol
        symb_b = self.atoms[rand_b].symbol
        system_changes = [(rand_a,symb_a,symb_b),(rand_b,symb_b,symb_a)]
        #print (system_changes)
        new_energy = self.atoms._calc.calculate( self.atoms, ["energy"], system_changes )

        if ( verbose ):
            print(new_energy,self.current_energy)

        accept = False
        if new_energy < self.current_energy:
            self.current_energy = new_energy
            accept = True
        else:
            kT = self.T*units.kB
            energy_diff = new_energy-self.current_energy
            probability = np.exp(-energy_diff/kT)
            if ( np.random.rand() <= probability ):
                self.current_energy = new_energy
                accept = True
            else:
                # Reset the sytem back to original
                self.atoms[rand_a].symbol = symb_a
                self.atoms[rand_b].symbol = symb_b
                accept = False

        # TODO: Wrap this functionality into a cleaning object
        if ( hasattr(self.atoms._calc,"clear_history") and hasattr(self.atoms._calc,"undo_changes") ):
            # The calculator is a CE calculator which support clear_history and undo_changes
            if ( accept ):
                self.atoms._calc.clear_history()
            else:
                self.atoms._calc.undo_changes()

        if ( accept ):
            # Update the atom_indices
            self.atoms_indx[symb_a][selected_a] = rand_b
            self.atoms_indx[symb_b][selected_b] = rand_a
        else:
            system_changes = [(rand_a,symb_a,symb_a),(rand_b,symb_b,symb_b)] # No changes to the system

        # Execute all observers
        for entry in self.observers:
            interval = entry[0]
            if ( self.current_step%interval == 0 ):
                obs = entry[1]
                obs(system_changes)
        return self.current_energy,accept
