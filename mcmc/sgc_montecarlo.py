from mcmc import montecarlo as mc
from mcmc.mc_observers import SGCObserver
import numpy as np
from ase.units import kB

class SGCMonteCarlo( mc.Montecarlo ):
    def __init__( self, atoms, temp, indeces=None, symbols=None ):
        mc.Montecarlo.__init__( self, atoms, temp, indeces=indeces )
        if ( not symbols is None ):
            # Override the symbols function in the main class
            self.symbols = symbols
        self.averager = SGCObserver( self.atoms._calc, self, len(self.symbols)-1 )
        self.chem_pots = []

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

    def runMC( self, steps = 10, verbose = False, chem_potential=None ):
        if ( chem_potential is None ):
            ex_chem_pot = {
                "c1_1":-0.1,
                "c1_2":0.05
            }
            raise ValueError( "No chemicla potentials given. Has to be dictionary of the form {}".format(ex_chem_pot) )

        eci = self.atoms._calc.eci
        keys = chem_potential.keys()
        keys.sort()
        for key in keys:
            eci[key] -= chem_potential[key]
            self.chem_pots.append( chem_potential[key] )
        self.atoms._calc.update_ecis( eci )
        self.averager.reset()
        self.attach( self.averager )
        mc.Montecarlo.runMC( self, steps=steps, verbose=verbose )

    def get_thermodynamic( self ):
        quantities = {}
        quantities["singlets"] = self.averager.singlets/self.current_step
        quantities["chem_pots"] = self.chem_pots
        quantities["energy"] = self.averager.energy/self.current_step
        for i in range( len(quantities["chem_pots"]) ):
            quantities["energy"] += quantities["chem_pots"][i]*quantities["singlets"][i]

        quantities["heat_capacity"] = self.averager.energy_sq/self.current_step - quantities["energy"]**2 + \
                                      self.averager.singl_eng/self.current_step - quantities["energy"]*quantities["singlets"]
        quantities["heat_capacity"] /= (kB*self.T**2)
