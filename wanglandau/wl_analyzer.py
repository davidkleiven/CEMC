from ase import units
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

class WangLandauSGCAnalyzer( object ):
    def __init__( self, energy, dos, chem_pot, gs_energy ):
        """
        Object for analyzing thermodynamics from the Density of States in the
        Semi Grand Cannonical Ensemble
        """
        self.E = energy
        self.dos = dos
        self.E0 = gs_energy
        self.E0 = np.min(self.E)
        self.chem_pot = chem_pot
        #self.extend_dos_by_extraploation()

    def extend_dos_by_extraploation( self ):
        """
        Extends the DOS by fitting a linear curve to the smallest points
        """
        slope,interscept,rvalue,pvalue,stderr = linregress( self.E[:3], np.log(self.dos[:3]) )
        dE = self.E[1]-self.E[0]
        low_energies = np.arange(self.E0,self.E[0],dE)
        if ( len(low_energies) == 0 ):
            return

        low_energy_dos = np.exp(interscept+slope*low_energies )
        self.E = np.append(low_energies,self.E)
        self.dos = np.append(low_energy_dos,self.dos)

    def partition_function( self, T ):
        """
        Computes the partition function in the SGC ensemble
        """
        weight = np.abs(self.E0-np.min(self.E))/(self.E[1]-self.E[0])
        return np.sum( self.dos*self._boltzmann_factor(T) )

    def _boltzmann_factor( self, T ):
        """
        Returns the boltzmann factor
        """
        return np.exp( -(self.E-self.E0)/(units.kB*T) )

    def internal_energy( self, T ):
        """
        Computes the average energy in the SGC ensemble
        """
        return np.sum( self.E*self.dos*self._boltzmann_factor(T) )/self.partition_function(T)

    def heat_capacity( self, T ):
        """
        Computes the heat capacity in the SGC ensemble
        """
        e_mean = self.internal_energy(T)
        esq = np.sum(self.E**2 *self.dos*self._boltzmann_factor(T) )/self.partition_function(T)
        return (esq-e_mean**2)/(units.kB*T**2)

    def free_energy( self, T ):
        """
        The thermodynamic potential in the SGC ensemble
        """
        return -units.kB*T*np.log(self.partition_function(T)) + self.E0

    def plot_dos( self ):
        """
        Plots the density of states
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, np.log(self.dos), ls="steps" )
        ax.set_xlabel( "Energy (eV/atom)" )
        ax.set_ylabel( "Density of states" )
        return fig

    def plot_degree_of_contribution( self, temps ):
        """
        Gives an estimate plot on how much each bin contributes to the partition
        function for each contribution
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        low = None
        high = None
        for T in temps:
            dist = self.dos*self._boltzmann_factor(T)
            ax.plot(self.E, dist, label="T={}K".format(T), ls="steps" )
            new_low = min([dist[0],dist[-1]])
            if ( low is None or new_low < low ):
                low = new_low
        ax.set_ylim(ymin=low)
        ax.legend( loc="best", frameon=False )
        ax.set_xlabel( "Energy (eV)" )
        ax.set_ylabel( "Distribution" )
        ax.set_yscale("log")
        return fig
