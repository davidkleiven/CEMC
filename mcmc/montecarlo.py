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

    def __init__(self, atoms, temp, indeces=None ):
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
        self.mean_energy = 0.0
        self.energy_squared = 0.0

        # Some member variables used to update the atom tracker, only relevant for canonical MC
        self.rand_a = 0
        self.rand_b = 0
        self.selected_a = 0
        self.selected_b = 0

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

    def update_tracker( self, system_changes ):
        """
        Update the atom tracker
        """
        symb_a = system_changes[0][0]
        symb_b = system_changes[1][0]
        self.atoms_indx[symb_a][self.selected_a] = self.rand_b
        self.atoms_indx[symb_b][self.selected_b] = self.rand_a


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
        self.mean_energy = 0.0
        self.energy_squared = 0.0
        self.current_step = 0
        while( step < steps ):
            step += 1
            en, accept = self._mc_step( verbose=verbose )
            self.mean_energy += self.current_energy
            self.energy_squared += self.current_energy**2

            if ( time.time()-start > self.status_every_sec ):
                print ("%d of %d steps. %.2f ms per step"%(step,steps,1000.0*self.status_every_sec/float(step-prev)))
                prev = step
                start = time.time()
        return totalenergies

    def get_thermodynamic( self ):
        """
        Compute thermodynamic quantities
        """
        quantities = {}
        print (self.current_step)
        quantities["energy"] = self.mean_energy/self.current_step
        mean_sq = self.energy_squared/self.current_step
        quantities["heat_capacity"] = (mean_sq-quantities["energy"]**2)/(units.kB*self.T**2)
        return quantities

    def get_trial_move( self ):
        self.rand_a = self.indeces[np.random.randint(0,len(self.indeces))]
        self.rand_b = self.indeces[np.random.randint(0,len(self.indeces))]
        symb_a = self.symbols[np.random.randint(0,len(self.symbols))]
        symb_b = symb_a
        while ( symb_b == symb_a ):
            symb_b = self.symbols[np.random.randint(0,len(self.symbols))]

        Na = len(self.atoms_indx[symb_a])
        Nb = len(self.atoms_indx[symb_b])
        self.selected_a = np.random.randint(0,Na)
        self.selected_b = np.random.randint(0,Nb)
        self.rand_a = self.atoms_indx[symb_a][self.selected_a]
        self.rand_b = self.atoms_indx[symb_b][self.selected_b]

        # TODO: The MC calculator should be able to have constraints on which
        # moves are allowed. CE requires this some elements are only allowed to
        # occupy some sites
        symb_a = self.atoms[self.rand_a].symbol
        symb_b = self.atoms[self.rand_b].symbol
        system_changes = [(self.rand_a,symb_a,symb_b),(self.rand_b,symb_b,symb_a)]
        return system_changes

    def _mc_step(self, verbose = False ):
        """
        Make one Monte Carlo step by swithing two atoms
        """
        self.current_step += 1
        number_of_atoms = len(self.atoms)


        system_changes= self.get_trial_move()
        new_energy = self.atoms._calc.calculate( self.atoms, ["energy"], system_changes )

        if ( verbose ):
            print(new_energy,self.current_energy,new_energy-self.current_energy)

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
                for change in system_changes:
                    indx = change[0]
                    old_symb = change[1]
                    self.atoms[indx].symbol = old_symb
                #self.atoms[self.rand_a].symbol = symb_a
                #self.atoms[self.rand_b].symbol = symb_b
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
            self.update_tracker( system_changes )
        else:
            new_symb_changes = []
            for change in system_changes:
                new_symb_changes.append( (change[0],change[1],change[1]) )
            #system_changes = [(self.rand_a,symb_a,symb_a),(self.rand_b,symb_b,symb_b)] # No changes to the system

        # Execute all observers
        for entry in self.observers:
            interval = entry[0]
            if ( self.current_step%interval == 0 ):
                obs = entry[1]
                obs(system_changes)
        return self.current_energy,accept
