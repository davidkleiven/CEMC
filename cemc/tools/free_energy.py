import copy
import numpy as np
from scipy.integrate import trapz
from ase.units import kB

class FreeEnergy(object):
    """
    Class that computes the Free Energy in the Semi Grand Canonical Ensemble
    """

    def __init__(self, chemical_potential=None, singlets=None ):
        pass

    def get_sgc_energy( self, internal_energy, singlets, chemical_potential ):
        """
        Returns the SGC energy
        """
        sgc_energy = copy.deepcopy( internal_energy )
        for key in chemical_potential.keys():
            if ( len(singlets[key]) != len(sgc_energy) ):
                raise ValueError( "The singlets should have exactly the same length as the internal energy array!" )
            sgc_energy -= chemical_potential[key]*np.array( singlets[key] )
        return sgc_energy

    def free_energy_isochemical( self, T=None, sgc_energy=None, nelem=None ):
        """
        Computes the Free Energy by thermodynamic integration from the high temperature limit
        The line integration is performed along a line of constant chemical potential

        Paramters
        ----------
        sgc_energy - The energy in the Semi Grand Canonical Ensemble (E-\mu n). Should be
                     normalized by the number of atoms.
        nelem - Number of elements
        T - temperatures in Kelvin
        """
        if ( nelem is None ):
            raise ValueError( "The number of elements has to be specified!" )
        if ( sgc_energy is None ):
            raise ValueError( "The SGC energy has to be given!" )
        if ( T is None ):
            raise ValueError( "No temperatures given!" )

        if ( len(T) != len(sgc_energy) ):
            raise ValueError( "The temperature array has to be the same length as the sgc_energy array!" )
        high_temp_lim = -np.log(nelem)

        # Sort the value in descending order
        srt_indx = np.argsort(T)[::-1]

        T = np.array( [T[indx] for indx in srt_indx] )
        sgc_energy = np.array( [sgc_energy[indx] for indx in srt_indx])

        beta_phi_ref = -np.log(nelem) + sgc_energy[0]/(kB*T[0])

        integrand = sgc_energy/(kB*T**2)
        integral = [trapz(integrand[:i],x=T[:i]) for i in range(1,len(T))]
        integral = np.array(integral)

        beta_phi = beta_phi_ref - integral
        phi = beta_phi*kB*T[1:]
        res = {
            "temperature":T[1:],
            "free_energy":phi,
            "temperature_integral":integral,
            "order":srt_indx
        }
        return res

    def helmholtz_free_energy( self, free_energy, chemical_potential, singlets ):
        """
        Compute the Helmholtz free energy from the Grand Potential (Free Energy in the SGC ensemble)

        Parameters
        ----------
        free_energy - Free energy in the SGC ensemble
        chemical_potential - chemical potential
        singlets - Exepctation value of the singlet terms. Make sure that it is sorted correctly!
                   The highest temperatures appear first
        """
        helmholtz = np.zeros_like( free_energy )
        helmholtz[:] = free_energy
        for key in chemical_potential.keys():
            helmholtz += chemical_potential[key]*np.array( singlets[key][1:] )
        return helmholtz
