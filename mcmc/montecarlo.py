# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division

import numpy as np
import ase.units as units
from wanglandau import ce_calculator
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
        self.current_energy = self.atoms.get_potential_energy() # Get starting energy

        totalenergies = []
        totalenergies.append(self.current_energy)
        for step in range(steps):
            en, accept = self._mc_step( verbose=verbose )
            if ( verbose ):
                print(accept)
            totalenergies.append(en)

        return totalenergies


    def _mc_step(self, verbose = False ):
        """ Make one Monte Carlo step by swithing two atoms """
        number_of_atoms = len(self.atoms)

        rand_a = self.indeces[np.random.randint(0,len(self.indeces))]
        rand_b = self.indeces[np.random.randint(0,len(self.indeces))]
        # At the moment rand_a and rand_b could be the same atom

#        rand_b = np.random.randint(0,number_of_atoms)
        #while (rand_a == rand_b):
        #    rand_b = np.random.randint(0,number_of_atoms)

        # TODO: The MC calculator should be able to have constraints on which
        # moves are allowed. CE requires this some elements are only allowed to
        # occupy some sites
        symb_a = self.atoms[rand_a].symbol
        symb_b = self.atoms[rand_b].symbol
        self.atoms[rand_a].symbol = symb_b
        self.atoms[rand_b].symbol = symb_a
        system_changes = [(rand_a,symb_a,symb_b),(rand_b,symb_b,symb_a)]
        new_energy = self.atoms.calculate( self.atoms, ["energy"], system_changes )
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
            if np.random.rand() <= probability:
                self.current_energy = new_energy
                accept = True
            else:
                # Reset the sytem back to original
                self.atoms[rand_a].symbol,self.atoms[rand_b].symbol = self.atoms[rand_b].symbol,self.atoms[rand_a].symbol

        for entry in self.observers:
            interval = entry[0]
            if ( self.current_step%interval == 0 ):
                obs = entry[1]
                if ( accept ):
                    obs(rand_a,rand_b)
                else:
                    obs(rand_a,rand_a)

        if ( isinstance(self.atoms._calc,ce_calculator.CE) ):
            if ( accept ):
                self.atoms._calc.clear_history()
            else:
                self.atoms._calc.undo_changes()
        return self.current_energy,accept
