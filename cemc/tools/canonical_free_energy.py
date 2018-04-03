import numpy as np
from ase.units import kB

class CanonicalFreeEnergy( object ):
    def __init__( self, composition ):
        self.comp = composition

    def reference_energy( self, T, energy ):
        """
        Computes the reference energy
        """
        concs = np.array( [value for key,value in self.comp.iteritems()] )
        infinite_temp_value = np.sum( concs*np.log(concs) )
        beta = kB*T
        ref_energy = infinite_temp_value + energy*beta
        return ref_energy

    def sort( self, temperature, internal_energy ):
        """
        Sort the value such that the largerst temperature goes first
        """
        srt_indx = np.argsort(temperature)[::-1]
        temp_srt = [temperature[indx] for indx in srt_indx]
        energy_srt = [internal_energy[indx] for indx in srt_indx]

        assert( temp_srt[0] > temp_srt[1] ) # Make sure the sorting is correct
        return np.array(temp_srt), np.array( energy_srt)

    def get( self, temperature, internal_energy ):
        """
        Compute the Helholtz Free Energy
        """
        temperature, internal_energy = self.sort( temperature, internal_energy )
        betas = 1.0/(kB*temperature)
        ref_energy = self.reference_energy( temperature[0], internal_energy[0] )

        free_energy = [np.trapz(internal_energy[:i],x=betas[:i]) for i in range(1,len(betas))]
        free_energy.append( np.trapz(internal_energy,x=betas) )
        return temperature, internal_energy, free_energy
