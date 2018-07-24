from ase.units import kB
import copy
import numpy as np

class LinearVibCorrection(object):
    def __init__( self, eci_per_kbT ):
        self.eci_per_kbT = eci_per_kbT
        self.orig_eci = None
        self.current_eci = None
        self.vibs_included = False
        self.temperature = 0.0

    def check_provided_eci_match( self, provided_eci ):
        """
        Check that the provided ecis match the ones stored in this object
        """
        if ( self.current_eci is None ):
            return
        return
        for key in self.current_eci.keys():
            if ( key not in provided_eci.keys() ):
                raise ValueError( "The keys don't match! Keys should be: {}. Keys provided: {}".format(self.current_eci.keys(),provided_eci.keys()) )

        for key in self.current_eci.keys():
            if ( not np.isclose( self.current_eci[key],provided_eci[key] ) ):
                raise ValueError( "Provided ECIs do not match the ones stored. Stored: {}. Provided: {}".format(self.current_eci,provided_eci))

    def include( self, eci_with_out_vib, T ):
        """
        Adds the vibrational ECIs to the orignal
        """
        self.temperature = T
        if ( self.vibs_included ):
            return
        self.check_provided_eci_match( eci_with_out_vib )
        for key in self.eci_per_kbT.keys():
            if ( not key in eci_with_out_vib.keys() ):
                raise KeyError( "The cluster {} is not in the original ECIs!".format(key) )

        for key,value in self.eci_per_kbT.items():
            eci_with_out_vib[key] += value*kB*T
        self.vibs_included = True
        self.current_eci = copy.deepcopy(eci_with_out_vib)
        return eci_with_out_vib

    def reset( self, eci_with_vibs ):
        """
        Removes the contribution from vibrations to the ECIs
        """
        self.check_provided_eci_match(eci_with_vibs)
        if ( not self.vibs_included ):
            return
        for key,value in self.eci_per_kbT.items():
            eci_with_vibs[key] -= value*kB*self.temperature
        self.current_eci = copy.deepcopy(eci_with_vibs)
        self.vibs_included = False
        return eci_with_vibs

    def energy_due_to_vibrations( self, T, cf ):
        """
        Computes the contribution to the energy due to internal vibrations
        """
        E = 0.0
        for key,value in self.eci_per_kbT:
            E += value*cf[key]*kB*T
        return E
