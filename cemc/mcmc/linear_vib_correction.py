from ase.units import kB
import copy

class LinearVibCorrection(object):
    def __init__( self, eci_per_kbT ):
        self.eci_per_kbT = eci_per_kbT
        self.T = 0.0
        self.eci_included = False

    def include( self, eci_with_out_vib, T ):
        """
        Adds the vibrational ECIs to the orignal
        """
        if ( self.eci_included ):
            return eci_with_out_vib
        self.T = T
        for key in self.eci_per_kbT.keys():
            if ( not key in eci_with_out_vib.keys() ):
                raise KeyError( "The cluster {} is not in the original ECIs!".format(key) )

        self.orig_eci = copy.deepcopy(eci_with_out_vib)
        for key,value in self.eci_per_kbT.iteritems():
            eci_with_out_vib[key] += value*kB*T
        self.eci_included = True
        return eci_with_out_vib

    def reset( self, eci_with_vib ):
        """
        Removes the contribution from vibrations to the ECIs
        """
        if ( not self.eci_included ):
            return eci_with_vib
        for key,value in self.eci_per_kbT.iteritems():
            eci_with_vib[key] -= value*kB*self.T
        self.eci_included = False
        return eci_with_vib

    def energy_due_to_vibrations( self, T, cf ):
        """
        Computes the contribution to the energy due to internal vibrations
        """
        E = 0.0
        for key,value in self.eci_per_kbT:
            E += value*cf[key]*kB*T
        return E
