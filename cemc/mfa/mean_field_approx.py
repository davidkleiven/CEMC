from ase.ce.settings import BulkCrystal
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.units import kB
import copy
import numpy as np

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

    def compute_partition_function_one_atom( self, indx, temperature ):
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

        beta = 1.0/(kB*temperature)
        return np.exp(-beta*E_sum)

    def partition_function( self, temperatures, chem_pot=None ):
        """
        Computes the partition function in the mean field approximation
        """
        if ( chem_pot is not None ):
            self.set_chemical_potentials( chem_pot )

        part_func = []
        for T in temperatures:
            Z = 1.0
            for i in range( len(self.bc.atoms) ):
                Z *= self.compute_partition_function_one_atom( i, T )
            part_func.append( Z )

        if ( chem_pot is not None ):
            self.reset_calculator_parameters()
        return part_func

    def free_energy( self, temperatures, chem_pot=None ):
        """
        Compute the free energy

        Returns
        --------
        Free energy in the Semi Grand Canonical Ensemble
        """
        Z = self.partition_function( temperatures, chem_pot=chem_pot )
        G = [self.E0-kB*T*z for (T,z) in zip(temperatures,Z)]
        return G
