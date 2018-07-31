from ase import units
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from cemc.wanglandau.wltools import get_formula
from scipy.stats import linregress

class WangLandauSGCAnalyzer( object ):
    def __init__( self, energy, dos, atomic_numbers, chem_pot=None ):
        """
        Object for analyzing thermodynamics from the Density of States in the
        Semi Grand Cannonical Ensemble
        """
        self.E = energy
        self.dos = dos
        self.E0 = np.min(self.E)
        self.chem_pot = chem_pot
        self.n_atoms = len(atomic_numbers)
        self.atomic_numbers = atomic_numbers
        self.poly_tail = np.zeros( len(self.E), dtype=np.uint8 )
        #self.extend_dos_by_extraploation()

    def normalize_dos_by_infinite_temp_limit( self ):
        """
        Normalize the DOS by using analytical expressions from infinite temperature
        """
        elm_count = {}
        for at_num in self.atomic_numbers:
            if ( at_num in elm_count.keys() ):
                elm_count[at_num] += 1
            else:
                elm_count[at_num] = 1
        sumDos = np.sum(self.dos)

        N = len(self.atomic_numbers)
        log_configs = N*np.log(N)-N
        for key,value in elm_count.items():
            log_configs -= (value*np.log(value)-value)
        factor = np.exp( log_configs - np.log(sumDos) )
        self.dos *= factor

    def get_chemical_formula( self ):
        """
        Returns the chemical formula of the object (only relevant if in cannonical ensemble)
        """
        return get_formula( self.atomic_numbers )


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
        return np.sum( self.E*self.dos*self._boltzmann_factor(T) )/(self.partition_function(T)*self.n_atoms)

    def heat_capacity( self, T ):
        """
        Computes the heat capacity in the SGC ensemble
        """
        e_mean = np.sum(self.E *self.dos*self._boltzmann_factor(T) )/(self.partition_function(T))
        esq = np.sum(self.E**2 *self.dos*self._boltzmann_factor(T) )/(self.partition_function(T))
        return (esq-e_mean**2)/(self.n_atoms*units.kB*T**2)

    def free_energy( self, T ):
        """
        The thermodynamic potential in the SGC ensemble
        """
        return (-units.kB*T*np.log(self.partition_function(T)) + self.E0)/self.n_atoms

    def entropy( self, T ):
        """
        Computes the entropy of the system
        """
        F = self.free_energy(T)
        U = self.internal_energy(T)
        S = (U-F)/T
        return S

    def plot_dos( self, fit="none", fig=None ):
        """
        Plots the density of states
        """
        x = 1000.0*self.E/self.n_atoms
        if ( fig is None ):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        else:
            ax = fig.axes[0]

        if ( np.sum(self.poly_tail) == 0 ):
            ax.plot( x, np.log(self.dos), ls="steps" )
        else:
            logdos = np.log(self.dos)
            data_c = logdos[self.poly_tail==0].tolist()
            x_c = x[self.poly_tail==0].tolist()
            ax.plot( x_c, data_c, ls="steps" )
            data_l = logdos[self.poly_tail==1].tolist()
            x_l = x[self.poly_tail==1].tolist()
            x_l.append(x_c[0])
            data_l.append(data_c[0])
            line = ax.plot( x_l,data_l, ls="steps" )
            data_r = logdos[self.poly_tail==2].tolist()
            x_r = x[self.poly_tail==2].tolist()
            data_r.insert(0,data_c[-1])
            x_r.insert(0,x_c[-1])
            ax.plot( x_r,data_r, ls="steps", color=line[-1].get_color() )
        ax.set_xlabel( "Energy (meV/atom)" )
        ax.set_ylabel( "Density of states" )

        if ( fit == "parabolic" ):
            fitted = self.parabolic_fit()
            ax.plot( x, fitted, label="Parabolic" )
        return fig

    def polynomial_tails( self, order=3, fraction=0.1 ):
        """
        Fits a power law to the log g(E) to extrapolate the tails
        """

        # Left tail
        logdos = np.log( self.dos )
        N = int( fraction*len(logdos) )
        if ( order >= N ):
            order = N-1
        data1 = logdos[:N]
        E1 = self.E[:N]
        M = np.zeros( (len(data1),order+1) )
        for n in range(order+1):
            M[:,n] = E1**n
        x_left, residual, rank, s = np.linalg.lstsq( M, logdos[:N] )

        # Right tail
        data2 = logdos[-N:]
        E2 = self.E[-N:]
        for n in range(order+1):
            M[:,n] = E2**n
        x_right, residual, rank, s = np.linalg.lstsq( M, logdos[-N:] )
        return x_left, x_right

    def update_dos_with_polynomial_tails( self, factor_low=1.05,factor_high=1.05, order=3, fraction=0.1 ):
        """
        Updates the DOS by using polynomial tails
        """
        x_left, x_right = self.polynomial_tails( order=order, fraction=fraction )
        dE = self.E[1]-self.E[0]
        center = 0.5*(self.E[-1]+self.E[0])
        width = self.E[-1] - self.E[0]
        Emin = center-0.5*factor_low*width
        Emax = center+0.5*factor_high*width
        N = int( (Emax-Emin)/dE )
        E = np.linspace( Emin,Emax,N )
        old_Emin = self.E[0]
        old_Emax = self.E[-1]
        old_Nbins = len(self.E)
        new_dos = np.zeros(N)
        self.poly_tail = np.zeros( N, dtype=np.uint8 )
        start = 0
        for i in range(N):
            if ( E[i] <= old_Emin ):
                new_dos[i] = np.polyval( x_left[::-1], E[i] )
                self.poly_tail[i] = 1
                start = i
            elif ( E[i] >= old_Emax ):
                new_dos[i] = np.polyval( x_right[::-1], E[i] )
                self.poly_tail[i] = 2
            else:
                indx = int( (E[i]-old_Emin)*old_Nbins/(old_Emax-old_Emin) )
                new_dos[i] = np.log(self.dos[i-start])
        self.E = E
        self.dos = np.exp(new_dos)
        self.E0 = np.min( self.E )

    def parabolic_fit( self ):
        """
        Fit a parabolic function to the the log DOS
        """
        logdos = np.log(self.dos)
        M = np.zeros( (len(logdos),3) )
        M[:,0] = 1.0
        M[:,1] = self.E
        M[:,2] = self.E**2
        x, residual, rank, s = np.linalg.lstsq( M, logdos )
        fitted = x[0] + x[1]*self.E + x[2]*self.E**2
        return fitted


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
            dist = np.log( self.dos*self._boltzmann_factor(T) )
            ax.plot(self.E, dist, label="T={}K".format(T), ls="steps" )
            new_low = min([dist[0],dist[-1]])
            if ( low is None or new_low < low ):
                low = new_low
        ax.set_ylim(ymin=low)
        ax.legend( loc="best", frameon=False )
        ax.set_xlabel( "Energy (eV)" )
        ax.set_ylabel( "log Occupational prob." )
        return fig
