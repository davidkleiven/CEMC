from ase.units import kB
import montecarlo as mc
import numpy as np

class SimmulatedAnnealingCanonical( object ):
    def __init__( self, atoms, temperatures, mode="minimize" ):
        self.atoms = atoms
        self.temperatures = temperatures
        self.mode = mode
        self.extremal_energy = 0.0

    def run( self, steps_per_temp=10000 ):
        if ( self.mode == "minimize" ):
            self.extremal_energy = np.inf
        elif ( self.mode == "maximize" ):
            self.extremal_energy = -np.inf
        for T in self.temperatures:
            print ("Current temperature {}K".format(T) )
            if ( self.mode == "minimize" ):
                mcobj = mc.Montecarlo( self.atoms, T )
            elif ( self.mode == "maximize" ):
                mcobj = mc.Montecarlo( self.atoms, -T )
            else:
                raise ValueError( "Unknown mode" )

            step = 0
            while( step < steps_per_temp ):
                step += 1
                energy, accept = mcobj._mc_step()
                if ( self.mode == "minimize" ):
                    self.extremal_energy = min([self.extremal_energy,energy])
                elif( self.mode == "maximize" ):
                    self.extremal_energy = max([self.extremal_energy,energy])
