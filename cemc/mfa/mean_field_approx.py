from ase.ce.settings import BulkCrystal
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.units import kB, kJ, mol
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

        self.flip_energies = []


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

    def compute_single_flip_energies( self, indx ):
        """
        Computes the energies corresponding to flipping atom at indx
        """
        orig_symbol = self.bc.atoms[indx].symbol
        flip_energies = []
        for symb in self.symbols:
            self.bc.atoms[indx].symbol = symb
            dE =  self.bc.atoms.get_potential_energy()-self.E0
            if ( dE < -1E-6 ):
                raise RuntimeError( "The reference structure should be a ground state! dE < 0.0 should not be possible. dE={}".format(dE) )
            flip_energies.append( self.bc.atoms.get_potential_energy()-self.E0 )
        self.bc.atoms[indx].symbol = orig_symbol
        return flip_energies

    def compute_flip_energies( self ):
        """
        Computes the flip energies for all the atoms
        """
        self.flip_energies = [self.compute_single_flip_energies(indx) for indx in range(len(self.bc.atoms))]

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
        self.bc.atoms._calc.atoms = None # Force a new energy calculation
        self.E0 = self.bc.atoms.get_potential_energy()

    def reset_calculator_parameters( self ):
        """
        Resets the ecis to their original value
        """
        self.bc.atoms._calc.eci = self.eci

    def compute_partition_function_one_atom( self, indx, beta ):
        """
        Computes the contribution to the partition function from one atom
        """
        Z = 0.0
        for E in self.flip_energies[indx]:
            Z += np.exp( -beta*E )
        return Z

    def partition_function( self, betas, chem_pot=None ):
        """
        Computes the partition function in the mean field approximation
        """
        if ( chem_pot is not None ):
            self.reset_calculator_parameters()
            self.set_chemical_potentials( chem_pot )
            self.E0 = self.bc.atoms.get_potential_energy()
            if ( (self.last_chem_pot) is None or (chem_pot!=self.last_chem_pot) ):
                self.flip_energies = []
                self.compute_flip_energies()
            self.last_chem_pot = chem_pot
        else:
            self.E0 = self.bc.atoms.get_potential_energy()
            if ( len(self.flip_energies) == 0 ):
                self.compute_flip_energies()
        part_func = []
        self.betas = betas
        for beta in betas:
            Z = 1.0
            for i in range( len(self.bc.atoms) ):
                Z *= self.compute_partition_function_one_atom( i, beta )
            part_func.append( Z )

        if ( chem_pot is not None ):
            self.reset_calculator_parameters()
        self.Z = part_func
        return self.Z

    def sort_data(self):
        """
        Sorts the data according to the betas
        """
        srt_indx = np.argsort( self.betas )
        self.betas = [self.betas[indx] for indx in srt_indx]
        self.Z = [self.Z[indx] for indx in srt_indx]


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
        betas = np.array( betas )
        Z = self.partition_function( betas, chem_pot=chem_pot )
        z = np.array(self.Z)
        kT = 1.0/betas
        G = self.E0 - kT*np.log(z)
        return np.array(G)/len(self.bc.atoms)

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
        Z = self.partition_function( betas, chem_pot=chem_pot)
        lnz = np.log( np.array(Z) )
        lnz_interp = UnivariateSpline( self.betas, lnz, k=3, s=1 )
        energy_interp = lnz_interp.derivative()
        energy = -self.E0-energy_interp( np.array(betas) )

        cf = self.get_cf_dict()
        if ( chem_pot is not None ):
            for key in chem_pot.keys():
                energy += chem_pot[key]*cf[key]
        return np.array(energy)/(len(self.bc.atoms))

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
