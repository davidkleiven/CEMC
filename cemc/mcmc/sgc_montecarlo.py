import montecarlo as mc
from cemc.mcmc.mc_observers import SGCObserver
import numpy as np
from ase.units import kB
import copy

class SGCMonteCarlo( mc.Montecarlo ):
    def __init__( self, atoms, temp, indeces=None, symbols=None ):
        mc.Montecarlo.__init__( self, atoms, temp, indeces=indeces )
        if ( not symbols is None ):
            # Override the symbols function in the main class
            self.symbols = symbols
        self.averager = SGCObserver( self.atoms._calc, self, len(self.symbols)-1 )
        self.chem_pots = []
        self.chem_pot_names = []
        self.has_attached_avg = False

    def get_trial_move( self ):
        indx = np.random.randint( low=0, high=len(self.atoms) )
        old_symb = self.atoms[indx].symbol
        new_symb = old_symb
        while( new_symb == old_symb ):
            new_symb = self.symbols[np.random.randint(low=0,high=len(self.symbols))]
        system_changes = [(indx,old_symb,new_symb)]
        return system_changes

    def update_tracker( self, system_changes ):
        """
        Override the update of the atom tracker. The atom tracker is irrelevant in the semi grand canonical ensemble
        """
        pass

    def include_chemcical_potential_in_ecis( self, chem_potential, eci ):
        """
        Including the chemical potentials in the ecis
        """
        self.chem_pots = []
        self.chem_pot_names = []
        keys = chem_potential.keys()
        keys.sort()
        for key in keys:
            self.chem_pots.append( chem_potential[key] )
            self.chem_pot_names.append(key)
            eci[key] -= chem_potential[key]
        return eci

    def reset_eci_to_original( self, eci_with_chem_pot ):
        """
        Resets the ecis to their original value
        """
        for name,val in zip(self.chem_pot_names,self.chem_pots):
            eci_with_chem_pot[name] += val
        return eci_with_chem_pot

    def runMC( self, steps = 10, verbose = False, chem_potential=None ):
        self.chem_pots = []
        if ( chem_potential is None ):
            ex_chem_pot = {
                "c1_1":-0.1,
                "c1_2":0.05
            }
            raise ValueError( "No chemicla potentials given. Has to be dictionary of the form {}".format(ex_chem_pot) )

        eci = self.include_chemcical_potential_in_ecis( chem_potential, self.atoms._calc.eci )
        self.atoms._calc.update_ecis( eci )
        self.averager.reset()

        if ( not self.has_attached_avg ):
            self.attach( self.averager )
            self.has_attached_avg = True
        mc.Montecarlo.runMC( self, steps=steps, verbose=verbose )

        eci = self.reset_eci_to_original( eci )
        self.atoms._calc.update_ecis( eci )

    def get_thermodynamic( self ):
        N = self.averager.counter
        quantities = {}
        quantities["singlets"] = self.averager.singlets/N
        quantities["chem_pots"] = self.chem_pots
        quantities["energy"] = self.averager.energy/N
        for i in range( len(quantities["chem_pots"]) ):
            quantities["energy"] += quantities["chem_pots"][i]*quantities["singlets"][i]

        quantities["heat_capacity"] = self.averager.energy_sq/N - (self.averager.energy/N)**2 + \
                                      np.sum( self.averager.singl_eng/N - (self.averager.energy/N)*quantities["singlets"] )
        quantities["heat_capacity"] /= (kB*self.T**2)
        quantities["temperature"] = self.T
        return quantities
