from ase.units import kB
import copy

class LinearVibCorrection(object):
    def __init__( self, eci_per_kbT ):
        self.eci_per_kbT = eci_per_kbT
        self.orig_eci = None

    def include( self, eci_with_out_vib, T ):
        """
        Adds the vibrational ECIs to the orignal
        """
        for key in self.eci_per_kbT.keys():
            if ( not key in eci_with_out_vib.keys() ):
                raise KeyError( "The cluster {} is not in the original ECIs!".format(key) )

        self.orig_eci = copy.deepcopy(eci_with_out_vib)
        for key,value in self.eci_per_kbT.iteritems():
            eci_with_out_vib[key] += value*kB*T
        return eci_with_out_vib

    def remove( self ):
        """
        Removes the contribution from vibrations to the ECIs
        """
        return self.orig_eci

    def energy_due_to_vibrations( self, T, cf ):
        """
        Computes the contribution to the energy due to internal vibrations
        """
        E = 0.0
        for key,value in self.eci_per_kbT:
            E += value*cf[key]*kB*T
        return E
