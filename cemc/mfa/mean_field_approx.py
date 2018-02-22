from ase.ce.settings import BulkCrystal
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.units import kB
import copy
import numpy as np
from scipy.interpolate import UnivariateSpline

class MeanFieldApprox( object ):
    """
    Class to study a cluster expansion model in the low temperature
    limit using the Mean Field Approximation
    """
    def __init__( self, bc ):
        self.bc = bc
        if ( not isinstance(self.bc.atoms._calc,ClusterExpansion) ):
            raise TypeError( "The calculator of the atoms object of BulkCrystal has to be a ClusterExpansion calculator!" )
        self.symbols = None
        self.get_symbols()

        # Keep copies of the original ecis and cluster names
        self.cluster_names = copy.deepcopy( self.bc.atoms._calc.cluster_names )
        self.eci = copy.deepcopy( self.bc.atoms._calc.eci )
        self.E0 = self.bc.atoms.get_potential_energy()
        self.Z = None
        self.betas = None
        self.last_chem_pot = None

    def get_symbols( self ):
        """
        Create a list of all possible symbols
        """
        if ( self.symbols is not None ):
            return self.symbols

        symbs = []
        for atom in self.bc.atoms:
            if ( atom.symbol in symbs ):
                continue
            symbs.append( atom.symbol )

        # Update the symbols
        self.symbols = symbs
        return self.symbols

    def set_chemical_potentials( self, chem_pot ):
        """
        Sets the chemical potential by subtracting the corresponding value from
        the single point terms in the ECIs of the calculator
        """
        for key,mu in chem_pot.iteritems():
            try:
                indx = self.bc.atoms._calc.cluster_names.index(key)
                self.bc.atoms._calc.eci[indx] -= mu
            except:
                pass

    def reset_calculator_parameters( self ):
        """
        Resets the ecis to their original value
        """
        self.bc.atoms._calc.eci = self.eci

    def compute_partition_function_one_atom( self, indx, beta ):
        """
        Computes the contribution to the partition function from one atom
        """
        orig_symbol = self.bc.atoms[indx].symbol
        E_sum = 0.0
        for symb in self.symbols:
            self.bc.atoms[indx].symbol = symb
            E_sum += (self.bc.atoms.get_potential_energy()-self.E0)

        # Set back to the original
        self.bc.atoms._calc.restore()

        return np.exp(-beta*E_sum)

    def partition_function( self, betas, chem_pot=None ):
        """
        Computes the partition function in the mean field approximation
        """
        if ( chem_pot is not None ):
            self.set_chemical_potentials( chem_pot )

        part_func = []
        for beta in betas:
            Z = 1.0
            for i in range( len(self.bc.atoms) ):
                Z *= self.compute_partition_function_one_atom( i, beta )
            part_func.append( Z )

        if ( chem_pot is not None ):
            self.reset_calculator_parameters()
        return part_func

    def sort_data(self):
        """
        Sorts the data according to the betas
        """
        srt_indx = np.argsort( self.betas )
        self.betas = [self.betas[indx] for indx in srt_indx]
        self.Z = [self.Z[indx] for indx in srt_indx]

    def update_partition_function( self, betas, chem_pot ):
        """
        Checks if there are new beta value if so update the partition function for those values
        """
        if ( self.betas is None ):
            self.Z = self.partition_function( betas, chem_pot=chem_pot )
            self.betas = betas
            self.last_chem_pot = chem_pot
            self.sort_data()
            return

        if ( chem_pot is not None ):
            if ( chem_pot != self.last_chem_pot ):
                self.Z = self.partition_function( betas, chem_pot=chem_pot )
                self.betas = betas
                self.last_chem_pot = chem_pot
                self.sort_data()
                return

        for beta in betas:
            if ( beta in self.betas ):
                continue
            self.Z.append( self.partition_function(beta, chem_pot=self.chem_pot) )
            self.betas.append(beta)
        self.sort_data()

    def free_energy( self, betas, chem_pot=None ):
        """
        Compute the free energy

        Parameters
        ----------
        betas - list of inverse temparatures (1/(kB*T))

        Returns
        --------
        Free energy in the Semi Grand Canonical Ensemble
        """
        self.update_partition_function( betas, chem_pot )
        G = [self.E0-z/beta for (beta,z) in zip(betas,self.Z)]
        return G

    def get_cf_dict( self ):
        """
        Returns the correlation function as a dictionary
        """
        cf = self.bc.atoms._calc.cf
        cf_dict = {cname:cfunc for cname,cfunc in zip(self.cluster_names,cf)}
        return cf_dict

    def internal_energy( self, betas, chem_pot=None ):
        """
        Compute the internal energy by computing the partial derivative
        with respect to beta
        """
        self.update_partition_function( betas, chem_pot )
        lnz = np.log( np.array(self.Z) )
        lnz_interp = UnivariateSpline( self.betas, lnz, k=3, s=1 )
        energy_interp = lnz_interp.derivative()
        energy = self.E0-energy_interp( np.array(betas) )

        cf = self.get_cf_dict()
        if ( chem_pot is not None ):
            for key in chem_pot.keys():
                energy += chem_pot[key]*cf[key]
        return energy

    def heat_capacity( self, betas, chem_pot=None ):
        """
        Computes the heat capacity by computing the derivative of the internal energy
        with respect to temperature
        """
        energy = self.internal_energy( betas, chem_pot=chem_pot )
        energy_interp = UnivariateSpline( betas, energy, k=3, s=1 )
        Cv_interp = energy_interp.derivative()
        Cv = -kB*np.array(betas**2)*Cv_interp( np.array(betas) )
        return Cv
