from cemc.mfa.mean_field_approx import MeanFieldApprox
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.units

class PhaseBounaryTracker(object):
    def __init__( self, gs1, gs2, mu_name="c1_1" ):
        """
        Class for tracker phase boundary
        """
        self.gs1 = gs1
        self.gs2 = gs2
        self.mu_name = mu_name
        self.check_gs()

        # Set the calculators of the atoms objects
        calc1 = ClusterExpansion( self.gs1["bc"], cluster_name_eci=self.gs1["eci"], init_cf=self.gs1["cf"] )
        calc2 = ClusterExpansion( self.gs2["bc"], cluster_name_eci=self.gs2["eci"], init_cf=self.gs2["cf"] )
        self.gs1["bc"].atoms.set_calculator( calc1 )
        self.gs2["bc"].atoms.set_calculator( calc2 )

        self.mfa1 = MeanFieldApprox( self.gs1["bc"] )
        self.mfa2 = MeanFieldApprox( self.gs2["bc"] )

    def get_zero_temperature_mu_boundary( self ):
        """
        Computes the chemical potential at which the two phases coexists
        at zero kelvin
        """

        # Assuming that the chemical potential at this point is not included into the ECIs
        E1 = self.gs1["bc"].atoms.get_potential_energy()
        E2 = self.gs1["bc"].atoms.get_potential_energy()
        x1 = self.gs1["cf"][self.mu_name]
        x2 = self.gs2["cf"][self.mu_name]
        mu_boundary = (E2-E1)/(x1-x2)
        return mu_boundary

    def mean_field_common_tangent_construction( self, T, relative_width=0.1, n_points=100 ):
        """
        Construction of a common tangent point
        """
        mu0 = self.get_zero_temperature_mu_boundary()

    def get_mean_field_phase_boundary( self, Tmin, Tmax, ntemps ):
        """
        Computes the phase boundary in the Mean Field Approximation
        """
        betas = np.linspace( 1.0/(kB*Tmin), 1.0/(kB*Tmax), ntemps )
        mu0 = self.get_zero_temperature_mu_boundary()
        db = betas[1]-betas[0]
        delta_singlets = self.gs1["cf"][self.mu_name] - self.gs2["cf"][self.mu_name] # Constant in MFA
        chem_pot = {self.mu_name:mu0}

    def check_gs( self ):
        """
        Check that provided parameters are correct
        """
        required_keys = ["bc","cf","eci"]
        for req_key in required_keys:
            if ( not req_key in self.gs1.keys() or not req_key in self.gs2.keys() ):
                raise ValueError( "Keys of the dictionaries describing the ground state are wrong. Has to include {}".format(required_keys) )

        if ( not self.mu_name in self.gs1["eci"].keys() or not self.mu_name in self.gs2["eci"].keys() ):
            raise ValueError( "There are no ECI corresponding to the chemical potential under consideration!" )
