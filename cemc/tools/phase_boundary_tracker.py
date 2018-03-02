from cemc.mfa.mean_field_approx import MeanFieldApprox
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.units import kB
from ase.visualize import view
import numpy as np
from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
from cemc.wanglandau.ce_calculator import CE
from scipy import stats
import copy
from cemc.tools.sequence_predicter import SequencePredicter

class PhaseBoundaryTracker(object):
    def __init__( self, gs1, gs2, mu_name="c1_1", chemical_potential=None ):
        """
        Class for tracker phase boundary
        """
        self.gs1 = gs1
        self.gs2 = gs2
        self.mu_name = mu_name
        if ( chemical_potential is None ):
            self.chemical_potential = {mu_name:0.0}
        else:
            self.chemical_potential=chemical_potential
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
        E2 = self.gs2["bc"].atoms.get_potential_energy()
        x1 = self.gs1["cf"][self.mu_name]
        x2 = self.gs2["cf"][self.mu_name]
        mu_boundary = (E2-E1)/(x2-x1)
        return mu_boundary/len( self.gs1["bc"].atoms )

    def is_equal( self, x1, x2, std1, std2, confidence_level=0.05 ):
        """
        Check if two numbers are equal provided that their standard deviations
        are known
        """
        diff = x2-x1
        std_diff = np.sqrt( std1**2 + std2**2 )
        if ( std_diff < 1E-4 ):
            # Cannot predict, say they are equal
            return True
        z_diff = diff/std_diff
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)
        if ( (z_diff < max_percentile) and (z_diff > min_percentile) ):
            return True
        return False

    def compositions_significantly_different( self, thermo1, thermo2, confidence_level=0.05, min_comp_res=0.01 ):
        """
        Returns True if the compositions are significantly different
        """
        name = "var_singlet_{}".format(self.mu_name)
        var1 = thermo1[name]/thermo1["n_mc_steps"]
        var2 = thermo2[name]/thermo2["n_mc_steps"]
        name = "singlet_{}".format(self.mu_name)
        comp1 = thermo1[name]
        comp2 = thermo2[name]
        diff = comp2-comp1
        std_diff = np.sqrt( var1+var2 )
        z = diff/std_diff
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)

        if ( np.abs(diff) < min_comp_res ):
            return False
        if (( z > max_percentile ) or ( z < min_percentile )):
            return True
        return False

    def bring_system_back_to_original_phase( self, sgc_mc, prev_mu=None, prev_comp=None ):
        """
        Brings the system back to the original phase by slowly varying
        """

    def separation_line( self, Tmin, Tmax, nsteps=10, n_mc_steps=100000 ):
        """
        Computes the separation line. Assuming that the zero kelvin line
        is a good approximation at Tmin
        """
        mu0 = self.get_zero_temperature_mu_boundary()
        calc1 = CE( self.gs1["bc"], self.gs1["eci"], initial_cf=self.gs1["cf"] )
        calc2 = CE( self.gs2["bc"], self.gs2["eci"], initial_cf=self.gs2["cf"] )
        self.gs1["bc"].atoms.set_calculator(calc1)
        self.gs2["bc"].atoms.set_calculator(calc2)
        sgc1 = SGCMonteCarlo( self.gs1["bc"].atoms, Tmin, symbols=["Al","Mg"] )
        sgc2 = SGCMonteCarlo( self.gs2["bc"].atoms, Tmin, symbols=["Al","Mg"] )
        beta = np.linspace( 1.0/(kB*Tmin), 1.0/(kB*Tmax), nsteps )
        mu = np.zeros(nsteps-1)
        mu[0] = mu0
        orig_mu = self.chemical_potential[self.mu_name]
        comp1 = np.zeros(nsteps)
        comp2 = np.zeros(nsteps)
        n = 1
        predicter = SequencePredicter( maxorder=10 )
        res = {
            "converged":False,
            "msg":"Simulations not started",
            "temperature":[],
            "singlet1":[],
            "singlet2":[],
            "mu":[]
        }
        while( n < nsteps ):
            print ("Current mu: {}. Current temperature: {}".format(mu[n-1],1.0/(kB*beta[n-1])))
            if ( n > 1 ):
                print ("Singlets: x1={},x2={}".format(comp1[n-2],comp2[n-2]))
            symbs1_old = [atom.symbol for atom in self.gs1["bc"].atoms]
            symbs2_old = [atom.symbol for atom in self.gs2["bc"].atoms]
            self.chemical_potential[self.mu_name] = mu[n-1]
            sgc1.T = 1.0/(kB*beta[n-1])
            sgc2.T = 1.0/(kB*beta[n-1])

            # Run Monte Carlo to sample
            sgc1.runMC( steps=n_mc_steps, equil=True, chem_potential=self.chemical_potential )
            thermo1 = sgc1.get_thermodynamic(reset_ecis=True)
            sgc2.runMC( steps=n_mc_steps, equil=True,  chem_potential=self.chemical_potential )
            thermo2 = sgc2.get_thermodynamic(reset_ecis=True)

            E2 = thermo2["energy"]
            E1 = thermo1["energy"]
            singl_name = "singlet_{}".format(self.mu_name)
            x2 = thermo2[singl_name]
            x1 = thermo1[singl_name]
            comp1[n-1] = x1
            comp2[n-1] = x2
            if ( not self.compositions_significantly_different(thermo1,thermo2,confidence_level=0.05) ):
                if ( n == 1 ):
                    raise RuntimeError( "One of the systems changed phase on first iteration. Verify that the chemical potentials are correct \
                                         and restart from a lower initial temperature" )
                diff1 = np.abs(comp1[n-1]-comp1[n-2])
                diff2 = np.abs(comp2[n-1]-comp2[n-2])
                if ( diff1 > diff2 ):
                    # System 1 changed phase
                    calc1.set_symbols( symbs1_old )
                else:
                    # System 2 changed phase
                    calc2.set_symbols( symbs2_old )

                # Check if the two phase boundaries met (check what the sequence of compositions predicts)
                x1_predict, cv = predicter( mu[:n-1], comp1[:n-1] )
                x2_predict, cv = predicter( mu[:n-1], comp2[:n-1] )
                var_name = "var_singlet_{}".format(self.mu_name)
                std1 = np.sqrt( thermo1[var_name]/thermo1["n_mc_steps"] )
                std2 = np.sqrt( thermo2[var_name]/thermo1["n_mc_steps"] )
                if ( self.is_equal(x1_predict,x2_predict,std1,std2,confidence_level=0.1) and (std1>1E-3) and (std2>1E-3)):
                    print (x1,x1_predict,std1)
                    print (x2,x2_predict,std2)
                    # The curves met
                    res["converged"] = True
                    res["msg"] = "Phase boundaries met"
                else:
                    # The current chemical potential is outside the region
                    # where the phases are metastable
                    # Set it to the average between this value and the previous
                    # where the phases are known to be metastable
                    mu[n-1] = 0.5*(mu[n-1]+mu[n-2])
            else:
                # Check if a third phase are the stable one
                if ( n > 2 ):
                    x1_predict, cv1 = predicter( mu[:n-1], comp1[:n-1] )
                    x2_predict, cv2 = predicter( mu[:n-1], comp2[:n-1] )
                    var_name = "var_singlet_{}".format(self.mu_name)
                    std1 = np.sqrt( thermo1[var_name]/thermo1["n_mc_steps"] )
                    std2 = np.sqrt( thermo2[var_name]/thermo1["n_mc_steps"] )
                    if ( not self.is_equal(x1,x1_predict,std1,std1) or not self.is_equal(x2,x2_predict,std2,std2) ):
                        # If the comositions are different, but the new composition failes from being
                        # predicted by the previous compositions
                        # a third phase must have appeared
                        print (x1,x1_predict,std1)
                        print (x2,x2_predict,std2)
                        res["converged"] = True
                        res["msg"] = "A third phase appears to be stable"

                # Both phases are metastable update the chemical potential
                # according to the differential equation
                # and proceed to the next temperature
                db = beta[n]-beta[n-1]
                rhs = (E2-E1)/( beta[n-1]*(x2-x1) ) - mu[n-1]/beta[n-1]
                mu[n] = mu[n-1] + rhs*db
                n += 1

            if ( res["converged"] ):
                res["temperature"] = 1.0/(kB*beta[:n])
                res["mu"] = mu[:n]
                res["singlet1"] = comp1[:n]
                res["singlet2"] = comp2[:n]
                return res

        res["temperature"] = 1.0/(kB*beta)
        res["mu"] = mu
        res["singlet1"] = comp1
        res["singlet2"] = comp2
        return res


    def mean_field_separation_line( self, Tmax, nsteps=10 ):
        """
        Computes the separation line in the mean field approximation
        """
        mu0 = self.get_zero_temperature_mu_boundary()
        Tmin = 1.0
        beta = np.linspace(1.0/(kB*Tmin), 1.0/(kB*Tmax), nsteps )
        mu = np.zeros(nsteps)
        mu[0] = mu0
        conc1 = np.zeros( nsteps )
        conc2 = np.zeros(nsteps)
        for n in range(1,nsteps):
            chem_pot = {self.mu_name:mu[n-1]}
            E2 = self.mfa2.internal_energy( [beta[n]], chem_pot=chem_pot )
            E1 = self.mfa1.internal_energy( [beta[n]], chem_pot=chem_pot )
            x2 = self.mfa2.average_singlets( [beta[n]], chem_pot=chem_pot )[self.mu_name]
            x1 = self.mfa1.average_singlets( [beta[n]], chem_pot=chem_pot )[self.mu_name]
            conc1[n] = x1
            conc2[n] = x2
            delta_b = beta[n]-beta[n-1]
            d_mu = (E2-E1)/(beta[n-1]*(x2-x1)) - mu[n-1]/beta[n-1]
            mu[n] = mu[n-1] + d_mu*delta_b
            print (1.0/(kB*beta[n]), mu[n])
        res = {
            "temperature":1.0/(kB*beta),
            "chemical_potential":mu,
            "singlet_1":conc1,
            "singlet_2":conc2
        }
        return res

    def mean_field_common_tangent_construction( self, T, relative_width=0.1, n_points=100 ):
        """
        Construction of a common tangent point
        """
        mu0 = self.get_zero_temperature_mu_boundary()
        mu_min = mu0 - relative_width*mu0/2.0
        mu_max = mu0 + relative_width/2.0
        mu = np.linspace( mu_min, mu_max, n_points )
        beta = 1.0/(kB*T)
        helmholtz_1 = []
        helmholtz_2 = []
        singlet_1 = []
        singlet_2 = []
        for i in range(len(mu)):
            mu_dict = {self.mu_name:mu[i]}
            singl_1 = self.mfa1.average_singlets( [beta], chem_pot=mu_dict )
            singl_2 = self.mfa2.average_singlets( [beta], chem_pot=mu_dict )
            F_1 = self.mfa1.helmholtz_free_energy( [beta], chem_pot=mu_dict )
            F_2 = self.mfa2.helmholtz_free_energy( [beta], chem_pot=mu_dict )
            singlet_1.append( singl_1[self.mu_name] )
            singlet_2.append( singl_2[self.mu_name] )
            helmholtz_1.append( F_1 )
            helmholtz_2.append( F_2 )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( singlet_1, helmholtz_1, "x" )
        ax.plot( singlet_2, helmholtz_2, "o", mfc="none" )
        plt.show()

    def get_mean_field_phase_boundary( self, Tmin, Tmax, ntemps ):
        """
        Computes the phase boundary in the Mean Field Approximation
        """
        betas = np.linspace( 1.0/(kB*Tmin), 1.0/(kB*Tmax), ntemps )
        mu0 = self.get_zero_temperature_mu_boundary()
        db = betas[1]-betas[0]
        delta_singlets = self.gs1["cf"][self.mu_name] - self.gs2["cf"][self.mu_name] # Constant in MFA

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
