import copy
import numpy as np
from scipy.integrate import trapz, simps
from ase.units import kB
from matplotlib import pyplot as plt

class FreeEnergy(object):
    """
    Class that computes the Free Energy in the Semi Grand Canonical Ensemble
    """

    def __init__(self, limit="hte", mfa=None ):
        allowed_limits = ["hte","lte"]
        if ( not limit in allowed_limits ):
            raise ValueError( "Limit has to be one of {}".format() )
        self.limit = limit
        self.mean_field = mfa
        self.chemical_potential = None

    def get_reference_beta_phi( self, temperature, sgc_energy, nelem=None ):
        """
        Returns the value of beta phi
        """
        if ( self.limit == "hte" ):
            data = {
                "temperature":temperature,
                "sgc_energy":sgc_energy
            }
            # Sort data such that the highest temperature appear first
            data = self.sort_key( data, mode="decreasing", sort_key="temperature" )
            beta_phi_ref = -np.log(nelem) + data["sgc_energy"][0]/(kB*data["temperature"][0])
            return beta_phi_ref
        else:
            if ( self.mean_field is None ):
                raise ValueError( "No mean field object has been set!" )
            data = {
                "temperature":temperature,
                "sgc_energy":sgc_energy
            }
            data = self.sort_key( data, mode="increasing", sort_key="temperature" )
            T_ref = data["temperature"][0]
            beta_ref = 1.0/(kB*T_ref)
            phi = self.mean_field.free_energy( [beta_ref], chem_pot=self.chemical_potential )
            return beta_ref*phi

    def sort_key( self, data, mode="decreasing", sort_key="x" ):
        """
        Sort data according to the values in x
        """
        allowed_modes = ["increasing", "decreasing"]
        if ( not mode in allowed_modes ):
            raise ValueError( "Mode has to be one of {}".format(allowed_modes) )

        if ( not sort_key in data.keys() ):
            raise ValueError( "Sort key not in dictionary!" )
        srt_indx = np.argsort( data[sort_key] )
        if ( mode == "decreasing" ):
            srt_indx = srt_indx[::-1]

        for key,value in data.iteritems():
            data[key] = np.array( [value[indx] for indx in srt_indx ] )

        # Make sure that the key_sort entries actually fits
        x = data[sort_key]
        for i in range(1,len(data[sort_key]) ):
            if ( mode == "decreasing" ):
                if ( x[i] > x[i-1] ):
                    raise ValueError( "The sequence should be decreasing!" )
            elif ( mode == "increasing" ):
                if ( x[i] < x[i-1] ):
                    raise ValueError( "The sequence should be increasing!" )
        return data

    def get_sgc_energy( self, internal_energy, singlets, chemical_potential ):
        """
        Returns the SGC energy
        """
        self.chemical_potential = chemical_potential
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
        """
        high_temp_lim = -np.log(nelem)

        # Sort the value in descending order
        srt_indx = np.argsort(T)[::-1]

        T = np.array( [T[indx] for indx in srt_indx] )
        sgc_energy = np.array( [sgc_energy[indx] for indx in srt_indx])
        """
        # Sort data
        data = {
            "temperature":T,
            "sgc_energy":sgc_energy
        }
        if ( self.limit == "hte" ):
            sort_mode = "decreasing"
        else:
            sort_mode = "increasing"
        data = self.sort_key( data, mode=sort_mode, sort_key="temperature" )
        T = data["temperature"]
        sgc_energy = data["sgc_energy"]
        #beta_phi_ref = -np.log(nelem) + sgc_energy[0]/(kB*T[0])
        beta_phi_ref = self.get_reference_beta_phi( T, sgc_energy, nelem=nelem )
        print (beta_phi_ref)
        beta = 1.0/(kB*T)
        integral = [trapz(sgc_energy[:i],x=beta[:i]) for i in range(1,len(beta))]
        integral.append(trapz(sgc_energy,x=beta))
        beta_phi = beta_phi_ref + integral
        phi = beta_phi*kB*T
        #phi = beta_phi
        res = {
            "temperature":T,
            "free_energy":phi,
            "temperature_integral":integral
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
