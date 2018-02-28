# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division

import numpy as np
import ase.units as units
from cemc.wanglandau import ce_calculator
import time
import logging
from mpi4py import MPI
from scipy import stats
import logging
#from ase.io.trajectory import Trajectory

class Montecarlo:
    """ Class for performing MonteCarlo sampling for atoms

    """

    def __init__(self, atoms, temp, indeces=None, mpicomm=None ):
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
        self.mpicomm = mpicomm
        self.logger = logging.getLogger( "MonteCarlo" )
        self.logger.setLevel( logging.DEBUG )
        ch = logging.StreamHandler()
        ch.setLevel( logging.INFO )
        self.logger.addHandler(ch)

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
        symb_a = system_changes[0][1]
        symb_b = system_changes[1][1]
        self.atoms_indx[symb_a][self.selected_a] = self.rand_b
        self.atoms_indx[symb_b][self.selected_b] = self.rand_a


    def attach( self, obs, interval=1 ):
        """
        Attach observers that is called on each MC step
        and receives information of which atoms get swapped
        """
        if ( callable(obs) ):
            self.observers.append( (interval,obs) )
        else:
            raise ValueError( "The observer has to be a callable class!" )

    def set_seeds(self):
        """
        This function guaranties different seeds on different processors
        """
        if ( self.mpicomm is None ):
            return

        rank = self.mpicomm.Get_rank()
        size = self.mpicomm.Get_size()
        maxint = np.iinfo(np.int32).max
        if ( rank == 0 ):
            seed = []
            for i in range(size):
                new_seed = np.random.randint(0,high=maxint)
                while( new_seed in seed ):
                    new_seed = np.random.randint(0,high=maxint)
                seed.append( new_seed )
        else:
            seed = None

        # Scatter the seeds to the other processes
        seed = self.mpicomm.scatter(seed, root=0)

        # Update the seed
        np.random.seed(seed)

    def equillibriate( self, window_length=1000, confidence_level=0.05, maxiter=1000 ):
        """
        Runs the MC until equillibrium is reached
        """
        E_prev = None
        var_E_prev = None
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)
        number_of_iterations = 1
        self.logger.info( "Equillibriating system" )
        self.logger.info( "Confidence level: {}".format(confidence_level))
        self.logger.info( "Percentiles: {}, {}".format(min_percentile,max_percentile) )
        self.logger.info( "{:10} {:10} {:10} {:10}".format("Energy", "std.dev", "delta E", "quantile") )
        for i in range(maxiter):
            number_of_iterations += 1
            self.mean_energy = 0.0
            self.energy_squared = 0.0
            for i in range(window_length):
                self._mc_step()
                self.mean_energy += self.current_energy
                self.energy_squared += self.current_energy**2
            E_new = self.mean_energy/window_length
            var_E_new = (self.energy_squared/window_length - E_new**2)/window_length

            if ( E_prev is None ):
                E_prev = E_new
                var_E_prev = var_E_new
                continue

            var_diff = var_E_new+var_E_prev
            diff = E_new-E_prev
            if ( var_diff < 1E-6 ):
                self.logger.info ( "Zero variance. System does not move." )
                z_diff = 0.0
            else:
                z_diff = diff/np.sqrt(var_diff)
            self.logger.info( "{:10.2f} {:10.6f} {:10.6f} {:10.2f}".format(E_new,var_E_new,diff, z_diff) )
            if( (z_diff < max_percentile) and (z_diff > min_percentile) ):
                self.logger.info( "System reached equillibrium in {} mc steps".format(number_of_iterations*window_length))
                self.mean_energy = 0.0
                self.energy_squared = 0.0
                self.current_step = 0
                return
            E_prev = E_new
            var_E_prev = var_E_new

        raise RuntimeError( "Did not manage to reach equillibrium!" )

    def runMC(self,steps = 10, verbose = False, equil=True ):
        """ Run Monte Carlo simulation

        Arguments:
        steps : Number of steps in the MC simulation

        """

        # Atoms object should have attached calculator
        # Add check that this is show
        self.current_energy = 1E8
        self._mc_step()
        #self.current_energy = self.atoms.get_potential_energy() # Get starting energy

        self.set_seeds()
        totalenergies = []
        totalenergies.append(self.current_energy)
        start = time.time()
        prev = 0
        self.mean_energy = 0.0
        self.energy_squared = 0.0
        self.current_step = 0

        if ( equil ):
            self.equillibriate()

        # self.current_step gets updated in the _mc_step function
        while( self.current_step < steps ):
            en, accept = self._mc_step( verbose=verbose )
            self.mean_energy += self.current_energy
            self.energy_squared += self.current_energy**2

            if ( time.time()-start > self.status_every_sec ):
                print ("%d of %d steps. %.2f ms per step"%(self.current_step,steps,1000.0*self.status_every_sec/float(self.current_step-prev)))
                prev = self.current_step
                start = time.time()
        return totalenergies

    def collect_energy( self ):
        """
        Sums the energy from each processor
        """
        if ( self.mpicomm is None ):
            return

        size = self.mpicomm.Get_size()
        recv = np.zeros(1)
        energy_arr = np.array(self.mean_energy)
        energy_sq_arr = np.array(self.energy_squared)
        self.mpicomm.reduce( energy_arr, recv, op=MPI.SUM, root= 0)
        self.mean_energy = recv[0]/size
        recv[0] = 0.0
        self.mpicomm.reduce( energy_sq_arr, recv, op=MPI.SUM, root=0 )
        self.energy_squared = recv[0]/size

    def get_thermodynamic( self ):
        """
        Compute thermodynamic quantities
        """
        self.collect_energy()
        quantities = {}
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
        if (new_energy < self.current_energy):
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
            system_changes = new_symb_changes
            #system_changes = [(self.rand_a,symb_a,symb_a),(self.rand_b,symb_b,symb_b)] # No changes to the system

        # Execute all observers
        for entry in self.observers:
            interval = entry[0]
            if ( self.current_step%interval == 0 ):
                obs = entry[1]
                obs(system_changes)
        return self.current_energy,accept
