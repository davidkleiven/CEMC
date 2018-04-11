import sys
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
import logging
from logging.handlers import RotatingFileHandler
from mpi4py import MPI
import h5py as h5
from scipy.interpolate import UnivariateSpline
import os
import gc

comm = MPI.COMM_WORLD

class PhaseChangedOnFirstIterationError(Exception):
    def __init__( self, message ):
        super( PhaseChangedOnFirstIterationError,self).__init__(message)

class PhaseBoundaryTracker(object):
    def __init__( self, gs1, gs2, mu_name="c1_0", chemical_potential=None, logfile="",
    conf_same_phase=0.45, conf_substep_conv=0.05, max_singlet_change=0.05, backupfile="backup_phase_track.h5" ):
        """
        Class for tracker phase boundary

        Parameters
        -----------
        gs1,gs2 - Dictionaries containing the following fields
                  {
                    "bc":BulkCrystal object,
                    "cf":Correlation function of the state of the atoms object in BulkCrystal
                    "eci":Effective Cluster Interactions
                  }
        mu_name - Name of the chemical potential to be altered.
                  If the system is binary the default: c1_0 is the correct one
        chemical_potential - Dictionary containing all the chemical potential
                             if the system under consideration is ternary this
                             looks like: {"c1_0":-1.06,"c1_1":-1.04}

                             If the system is binary this field can be left open
        logfile - Filename where log messages should be written
        conf_same_phase - Confidence level used for hypothesis testing when checking if the two systems
                          have ended up in the same phase. Has to be in the range [0.0,0.5)
        conf_substep_conv - Confidence level used for hypethesis testing when checking if the stepsize
                            needs to be refined. Has to be in the range[0.0,0.5)
        max_singlet_change - The maximum amount the singlet terms are allowed to change on each step
        """
        self.gs1 = gs1
        self.gs2 = gs2
        self.check_gs_argument(gs1)
        self.check_gs_argument(gs2)
        self.mu_name = mu_name
        self._conf_same_phase = conf_same_phase
        self._conf_substep_conv = conf_substep_conv
        self.max_singlet_change = max_singlet_change
        self.backupfile = backupfile

        if ( os.path.exists(self.backupfile) ):
            os.remove(self.backupfile)

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
        self.current_backup_indx = 0

        self.logger = logging.getLogger( "PhaseBoundaryTracker" )
        self.logger.setLevel( logging.INFO )
        if ( logfile == "" ):
            ch = logging.StreamHandler()
        else:
            ch = logging.FileHandler( logfile )
        if ( not self.logger.handlers ):
            self.logger.addHandler(ch)

    @property
    def conf_same_phase(self):
        return self._conf_same_phase

    @conf_same_phase.setter
    def conf_same_phase(self,value):
        if ( value >= 0.5 or value < 0.0 ):
            raise ValueError( "Confidence level has to be in the range [0.0,0.5). Given: {}".format(value) )
        self._conf_same_phase = value
    @property
    def conf_substep_conv(self):
        return self._conf_substep_conv

    @conf_substep_conv.setter
    def conf_substep_conv(self,value):
        if ( value >= 0.5 or value < 0.0 ):
            raise ValueError( "Confidence level has to be in the range [0.0,0.5). Given: {}".format(value) )
        self._conf_substep_conv = value

    def check_gs_argument( self, gs ):
        """
        Check that the gs arguments contain the correct fields
        """
        required_fields = ["bc","cf","eci"]
        keys = gs.keys()
        for key in keys:
            if ( key not in required_fields ):
                raise ValueError( "The GS argument has to contain {} keys. Given {}".format(required_fields,keys) )

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

    def log( self, msg, mode="info" ):
        """
        Print message for logging
        """
        rank = comm.Get_rank()
        if ( rank == 0 ):
            if ( mode == "info" ):
                self.logger.info( msg )
            elif ( mode == "warning" ):
                self.logger.warning(msg)

    def backup( self, data, dsetname="data" ):
        """
        Stores backup data to hdf5 file
        """
        rank = comm.Get_rank()
        if ( rank == 0 ):
            with h5.File( self.backupfile, 'a' ) as f:
                grp = f.create_group(dsetname+"{}".format(self.current_backup_indx))
                for key,value in data.iteritems():
                    grp.create_dataset( key, data=value )
            self.current_backup_indx += 1
        comm.barrier()

    def predict_composition( self, comp, temperatures, target_temp ):
        """
        Performs a prediction of the next composition value based on history
        """
        if ( len(comp) <= 2 ):
            return comp[-1]
        x = np.arange(0,len(temperatures))[::-1]
        weights = np.exp(-2.0*x/len(temperatures) )

        # Weight the data sich that the last point is more important than
        # the first.
        # Impact of the first point is exp(-2) relative to the impact of the
        # last point
        spl = UnivariateSpline( temperatures, comp, k=2 )
        predicted_comp = spl(target_temp)
        return predicted_comp

    def is_equal( self, x1, x2, std1, std2, confidence_level=0.05, stdtol=1E-6, eps_fallback=0.05 ):
        """
        Check if two numbers are equal provided that their standard deviations
        are known
        """
        if ( confidence_level >= 0.5 ):
            raise ValueError( "The confidence level has to be in the range [0,0.5)")

        diff = x2-x1
        std_diff = np.sqrt( std1**2 + std2**2 )

        if ( std_diff < stdtol ):
            # Standard deviation is close to zero. Use the eps_fallback to judge
            # if the two are equal
            return np.abs(x2-x1) < eps_fallback

        z_diff = diff/std_diff
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)
        if ( (z_diff < max_percentile) and (z_diff > min_percentile) ):
            return True
        return False

    def compositions_significantly_different( self, thermo1, thermo2, confidence_level=0.05 ):
        """
        Returns True if the compositions are significantly different
        """
        name = "var_singlet_{}".format(self.mu_name)
        var1 = thermo1[name]#/thermo1["n_mc_steps"]
        var2 = thermo2[name]#/thermo2["n_mc_steps"]
        name = "singlet_{}".format(self.mu_name)
        comp1 = thermo1[name]
        comp2 = thermo2[name]
        diff = comp2-comp1
        std_diff = np.sqrt( var1+var2 )
        z = diff/std_diff
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)

        if (( z > max_percentile ) or ( z < min_percentile )):
            return True
        return False

    def find_new_mu( self, sgc_mc, dmu, direction, current_comp, thermo, current_mu ):
        """
        Brings the system back to the original phase by slowly varying the chemical potential
        """
        changed_phase = False
        mu = current_mu
        self.logger.info( "*********** Finding new chemical potential *************" )
        print (thermo)
        while( not changed_phase ):
            if ( direction == "increase" ):
                mu += dmu
            else:
                mu -= dmu
            self.logger.info( "Current mu: {}".format(mu) )
            self.logger.handlers[0].flush()
            #self.flush_log()
            self.chemical_potential[self.mu_name] = mu
            sgc_mc.runMC( steps=100000, chem_potential=self.chemical_potential )
            thermo_new = sgc_mc.get_thermodynamic()
            print (thermo_new)
            changed_phase = self.compositions_significantly_different( thermo, thermo_new, confidence_level=0.05 )
            mu_phase_change = mu

        # Store the symbols
        symbs_old = [atom.symbol for atom in sgc_mc.atoms]

        # Run until the phase change back again
        changed_phase = False
        thermo = thermo_new
        self.logger.info( "*********** Find mu that makes the system change to the wrong phase ***************" )
        while( not changed_phase ):
            if ( direction == "increase" ):
                mu -= dmu # To come back to the wrong phase the chemical potential has to be increased
            else:
                mu += dmu
            self.logger.info( "Current mu: {}".format(mu))
            self.logger.handlers[0].flush()
            #self.flush_log()
            self.chemical_potential[self.mu_name] = mu
            sgc_mc.runMC( steps=100000, chem_potential=self.chemical_potential )
            thermo_new = sgc_mc.get_thermodynamic()
            changed_phase = self.compositions_significantly_different( thermo, thermo_new, confidence_level=0.05 )
            mu_phase_change_back = mu

        # Set the symbols to the correct phase
        sgc_mc.atoms._calc.set_symbols( symbs_old )
        return 0.5*( mu_phase_change + mu_phase_change_back )

    def on_similar_composition( self, args ):
        """
        Handling the case when the composition of the two "phases" have become
        equal
        """
        self.logger.info( "Composition of the two phases are similar!" )
        diff1 = np.abs( args["current_comp1"] - args["prev_comp1"] )
        diff2 = np.abs( args["current_comp2"] - args["prev_comp2"] )

        # Check if the two phase boundaries met (check what the sequence of compositions predicts)
        x1_pred = args["x1_pred"]
        x2_pred = args["x2_pred"]
        std1 = args["std1"]
        std2 = args["std2"]
        eq = self.is_equal( x1_pred, x2_pred, std1, std2, confidence_level=0.05 )
        #eq = (np.abs(x1_pred-x2_pred)<0.1)
        res = {"converged":False,"msg":"Continue simulation","mu":args["prev_mu"]}
        if ( eq ):
            #print (x1,x1_predict,std1)
            #print (x2,x2_predict,std2)
            # The curves met
            res["converged"] = True
            res["msg"] = "Phase boundaries met"
            return res

        # Simulations not converged, chemical potential is outside the region
        # where both phases are stable
        dmu = np.abs( args["prev_mu"]-args["current_mu"] )/args["n_mu_steps"]
        if ( diff1 > diff2 ):
            # System 1 changed phase
            self.logger.info( "System 1 changed phase" )
            new_mu = self.find_new_mu( args["sgc1"], dmu, "increase", args["current_comp1"], args["thermo1"], args["current_mu"] )
        else:
            self.logger.info( "System 2 changed phase" )
            new_mu = self.find_new_mu( args["sgc2"], dmu, "decrease", args["current_comp2"], args["thermo2"], args["current_mu"] )
        res["mu"] = new_mu
        return res

    def on_different_composition( self, args ):
        """
        Handling the case when the compositions are different
        """
        self.logger.info( "Compositions are different" )
        comp1_can_be_predicted = self.is_equal( args["current_comp1"], args["x1_pred"], args["std1"], args["std1"], confidence_level=0.0001 )
        comp2_can_be_predicted = self.is_equal( args["current_comp2"], args["x2_pred"], args["std2"], args["std2"], confidence_level=0.0001 )
        ##comp1_can_be_predicted = ( np.abs(args["current_comp1"]-args["x1_pred"] ) < 0.1 )
        ##comp2_can_be_predicted = ( np.abs(args["current_comp2"]-args["x2_pred"] ) < 0.1 )
        res = {"converged":False, "msg":"Continue simulation"}
        if ( not comp1_can_be_predicted and not comp2_can_be_predicted ):
            self.logger.info( "Both systems changed phase. Resetting mu to the previous value" )
            # Both systems changed phase --> impossible. Typically due to a
            # two large step in the chemical potential.
            # Reset mu to the previous value
            res["msg"] = "Reset mu"
            return res
        if ( not comp1_can_be_predicted or not comp2_can_be_predicted ):
            # Both compositions are different, but one of them cannot be
            # predicted by the sequence of compositions
            # One of the system went to a third phase
            print (args["current_comp1"],args["x1_pred"])
            print (args["current_comp2"],args["x2_pred"])
            res["converged"] = True
            res["msg"] = "One of the phases seems to have ended up in a third phase"
            return res
        return res

    def composition_first_larger_than_second( self, x1, x2 ):
        return x1 > x2

    def system_changed_phase( self, prev_comp, comp ):
        """
        Check if composition changes too much from one step to another
        """
        return np.abs( prev_comp-comp ) > self.max_singlet_change

    def composition_change_too_large( self, prev_comp, comp, std, threshold=0.05, confidence_level=0.45 ):
        """
        Check if the change in composition is larger than the allowed tolerance
        """
        diff = np.abs(prev_comp-comp)-threshold
        std_diff = np.sqrt(2.0)*std
        z_diff = diff/std_diff
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)
        if ( z_diff > max_percentile or z_diff < min_percentile ):
            return True
        return False

    def reset_symbols( self, calc, old_symbs ):
        """
        Resetting symbol and print a log message of the new elements
        """
        calc.set_symbols( old_symbs )
        cf = calc.get_cf()
        singlets = {}
        for key,value in cf.iteritems():
            if ( key.startswith("c1") ):
                singlets[key] = value
        self.log( "Singlets in system: {}".format(singlets) )


    def separation_line_adaptive_euler( self, T0=100, min_step=1, stepsize=100, mc_args={} ):
        """
        Solve the differential equation using adaptive euler
        """
        mu_prev = self.get_zero_temperature_mu_boundary()
        calc1 = CE( self.gs1["bc"], self.gs1["eci"], initial_cf=self.gs1["cf"] )
        calc2 = CE( self.gs2["bc"], self.gs2["eci"], initial_cf=self.gs2["cf"] )
        self.gs1["bc"].atoms.set_calculator(calc1)
        self.gs2["bc"].atoms.set_calculator(calc2)
        reference_logical_phase_check = False
        orig_stepsize = stepsize

        if ( "equil" in mc_args.keys() ):
            if ( not mc_args["equil"] ):
                self.log( "This scheme requires that the system can equillibriate. Changing to equil=True" )

        mc_args["equil"] = True

        if ( comm.Get_size() > 1 ):
            mpicomm = comm
        else:
            mpicomm = None
        sgc1 = SGCMonteCarlo( self.gs1["bc"].atoms, T0, symbols=["Al","Mg"], mpicomm=mpicomm )
        sgc2 = SGCMonteCarlo( self.gs2["bc"].atoms, T0, symbols=["Al","Mg"], mpicomm=mpicomm )
        comp1 = []
        comp2 = []
        mu_array = []
        temp_array = []
        dT = stepsize
        Tprev = T0
        target_comp1 = None
        target_comp2 = None
        target_temp = None
        rhs_prev = None
        is_first = True
        mu = mu_prev
        n_steps_required_to_reach_temp = 1
        substep_count = 1
        update_symbols1 = True
        update_symbols2 = True
        T = Tprev
        singl_name = "singlet_{}".format(self.mu_name)
        ref_compare = False
        iteration = 1
        symbs1_old = [atom.symbol for atom in self.gs1["bc"].atoms]
        symbs2_old = [atom.symbol for atom in self.gs2["bc"].atoms]
        while( stepsize > min_step ):
            self.log( "Current temperature {}K. Current chemical_potential: {} eV/atom".format(int(T),mu) )
            self.chemical_potential[self.mu_name] = mu
            mc_args["chem_potential"] = self.chemical_potential
            Tnext = T+dT

            beta = 1.0/(kB*T)
            sgc1.T = T
            sgc2.T = T
            comm.barrier()
            sgc1.runMC( **mc_args )
            thermo1 = sgc1.get_thermodynamic()
            comm.barrier()
            sgc2.runMC( **mc_args )
            thermo2 = sgc2.get_thermodynamic()

            x1 = thermo1[singl_name]
            x2 = thermo2[singl_name]
            E1 = thermo1["energy"]
            E2 = thermo2["energy"]
            rhs = (E2-E1)/( beta*(x2-x1)*len(sgc1.atoms) ) - mu/beta

            if ( is_first ):
                rhs_prev = rhs
                is_first = False
                beta_prev = 1.0/(kB*T)
                mu_prev = mu
                Tprev = T
                comp1.append(x1)
                comp2.append(x2)
                temp_array.append(T)
                mu_array.append(mu)
                T += dT
                beta_next = 1.0/(kB*T)
                dbeta = beta_next-beta_prev
                mu += rhs*dbeta
                ref_compare = self.composition_first_larger_than_second( x1, x2 )
                continue

            x1 = thermo1[singl_name]
            x2 = thermo2[singl_name]

            var_name = "var_singlet_{}".format(self.mu_name)
            std1 = np.sqrt( thermo1[var_name] )
            std2 = np.sqrt( thermo2[var_name] )
            eps = 0.01

            # Check if the two phases ended up in the same phase
            x1_equal_to_x2 = self.is_equal( x1, x2, std1, std2, confidence_level=0.45 )

            compositions_swapped = self.composition_first_larger_than_second(x1,x2) != ref_compare

            # Check if one of the systems changed phase
            if ( len(comp1) == 0 or len(comp2) == 0 ):
                ref1 = x1
                ref2 = x2
            else:
                ref1 = self.predict_composition( comp1, temp_array, T )
                ref2 = self.predict_composition( comp2, temp_array, T )
                self.log( "Temperatures used for prediction:")
                self.log( "{}".format(temp_array) )
                self.log( "Compositions syst1:" )
                self.log( "{}".format(comp1) )
                self.log( "Predicted composition for T={}K: {}".format(T,ref1))
                self.log( "Compositions syst2:" )
                self.log( "{}".format(comp2) )
                self.log( "Predicted composition for T={}K: {}".format(T,ref2) )

            one_system_changed_phase = self.system_changed_phase(x1,ref1) or self.system_changed_phase(x2,ref2)
            if ( x1_equal_to_x2 or compositions_swapped or one_system_changed_phase):
                if ( compositions_swapped ):
                    self.log( "Composition swapped" )
                elif ( one_system_changed_phase ):
                    self.log( "One of the systems changed phase" )
                else:
                    self.log( "Phase 1 and Phase 2 appears to be the same phase" )
                self.log( "Refine the step size and compute a new target temperature" )
                self.log( "Too reduce the chance of spurious phase change of this type happening again, the trial step size will also be reduced" )
                dT /= 2.0
                stepsize = dT # No point in trying larger steps than this
                T = Tprev
                mu = mu_prev
                rhs = rhs_prev

                self.reset_symbols( calc1, symbs1_old )
                self.reset_symbols( calc2, symbs2_old )

                self.log( "New step size: {}K".format(dT) )
                self.log( "New trial step size (used when computing the first target temperature after convergence): {}K".format(stepsize))
            else:
                target_comp1 = ref1
                target_comp2 = ref2

                # Compare with the reference composition
                x1_is_equal = self.is_equal( target_comp1, x1, std1, std1, stdtol=1E-6, eps_fallback=eps )
                x2_is_equal = self.is_equal( target_comp2, x2, std2, std2, stdtol=1E-6, eps_fallback=eps )

                if ( x1_is_equal and x2_is_equal ):
                    self.log( "Converged. Proceeding to the next temperature interval" )
                    # Converged
                    stepsize = orig_stepsize
                    dT = stepsize
                    Tprev = T
                    rhs_prev = rhs
                    mu_prev = mu
                    comp1.append(x1)
                    comp2.append(x2)
                    temp_array.append(T)
                    mu_array.append(mu)
                    backupdata = {
                        "comp1":comp1,
                        "comp2":comp2,
                        "temperature":temp_array,
                        "mu":mu_array
                    }
                    self.backup( backupdata, dsetname="iter")
                else:
                    # Did not converge reset and decrease the stepsize
                    T = Tprev
                    dT /= 2.0
                    mu = mu_prev
                    rhs = rhs_prev

                    # Update the target compositions to the new ones
                    self.log( "Did not converge. Updating target compositions. Refining stepsize. New stepsize: {}K".format(dT) )
                    self.log( "Resetting system" )

                    # Reset the system to pure phases
                    self.reset_symbols( calc1, symbs1_old )
                    self.reset_symbols( calc2, symbs2_old )

            beta_prev = 1.0/(kB*T)
            T += dT
            beta_next = 1.0/(kB*T)
            dbeta = beta_next - beta_prev
            mu += rhs*dbeta

            # Append the last step to the array
            if ( stepsize <= min_step ):
                temp_array.append(T)
                mu_array.append(mu)
                comp1.append(x1)
                comp2.append(x2)

        res = {}
        res["temperature"] = temp_array
        res["mu"] = mu_array
        res["singlet1"] = comp1
        res["singlet2"] = comp2
        res["msg"] = "Not able to make progress with the smalles stepsize {}K".format(min_step)
        return res

    def separation_line( self, temperatures, n_mc_steps=100000 ):
        """
        Computes the separation line. Assuming that the zero kelvin line
        is a good approximation at Tmin
        """
        nsteps = len(temperatures)
        Tmin = temperatures[0]
        Tmax = temperatures[-1]
        mu0 = self.get_zero_temperature_mu_boundary()
        calc1 = CE( self.gs1["bc"], self.gs1["eci"], initial_cf=self.gs1["cf"] )
        calc2 = CE( self.gs2["bc"], self.gs2["eci"], initial_cf=self.gs2["cf"] )
        self.gs1["bc"].atoms.set_calculator(calc1)
        self.gs2["bc"].atoms.set_calculator(calc2)
        sgc1 = SGCMonteCarlo( self.gs1["bc"].atoms, Tmin, symbols=["Al","Mg"], logfile="log_syst1.log" )
        sgc2 = SGCMonteCarlo( self.gs2["bc"].atoms, Tmin, symbols=["Al","Mg"], logfile="log_syst2.log" )
        beta = 1.0/(kB*temperatures)
        dbeta = beta[1]-beta[0]
        mu = np.zeros(nsteps)
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
            self.logger.info( "Current mu: {}. Current temperature: {}K".format(mu[n-1],1.0/(kB*beta[n-1])))
            self.logger.handlers[0].flush()
            #self.flush_log()
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
            self.logger.info("Singlets: x1={},x2={}".format(comp1[n-1],comp2[n-1]))
            self.logger.handlers[0].flush()

            var_name = "var_singlet_{}".format(self.mu_name)
            std1 = np.sqrt( thermo1[var_name]/thermo1["n_mc_steps"] )
            std2 = np.sqrt( thermo2[var_name]/thermo2["n_mc_steps"] )
            std1 = np.sqrt( thermo1[var_name] )
            std2 = np.sqrt( thermo2[var_name] )
            if ( not self.compositions_significantly_different(thermo1,thermo2,confidence_level=0.05) ):
                if ( n == 1 ):
                    raise PhaseChangedOnFirstIterationError( "One of the systems changed phase on first iteration. Verify that the chemical potentials are correct \
                                         and restart from a lower initial temperature" )
                args = {
                    "current_comp1":comp1[n-1],
                    "prev_comp1":comp1[n-2],
                    "current_comp2":comp2[n-1],
                    "prev_comp2":comp2[n-2],
                    "x1_pred":predicter( mu[:n], comp1[:n] )[0],
                    "x2_pred":predicter( mu[:n], comp2[:n] )[0],
                    "sgc1":sgc1,
                    "sgc2":sgc2,
                    "std1":std1,
                    "std2":std2,
                    "prev_mu":mu[n-2],
                    "current_mu":mu[n-1],
                    "n_mu_steps":10,
                    "thermo1":thermo1,
                    "thermo2":thermo2
                }
                result = self.on_similar_composition( args )
                res["converged"] = result["converged"]
                res["msg"] = result["msg"]
                mu[n-1] = result["mu"]
            else:
                x1_pred, cv1 = predicter( mu[:n], comp1[:n] )
                x2_pred, cv2 = predicter( mu[:n], comp2[:n] )
                args = {
                    "current_comp1":comp1[n-1],
                    "x1_pred":x1_pred,
                    "x2_pred":x2_pred,
                    "std1":std1,
                    "std2":std2,
                    "current_comp2":comp2[n-1]
                }
                result = self.on_different_composition( args )
                res["converged"] = result["converged"]
                update_mu_using_diff_equation = True
                if ( res["msg"] == "Reset mu" ):
                    mu[n-1] = mu[n-2]
                    calc1.set_symbols( symbs1_old )
                    calc2.set_symbols( symbs2_old )
                    update_mu_using_diff_equation = False
                else:
                    res["msg"] = result["msg"]

                if ( update_mu_using_diff_equation and not result["converged"] ):
                    db = beta[n]-beta[n-1]
                    #print (E2,E1,x2,x1)
                    #print ( (E2-E1)/((x2-x1)*len(sgc1.atoms)), mu[n-1])
                    rhs = (E2-E1)/( beta[n-1]*(x2-x1)*len(sgc1.atoms) ) - mu[n-1]/beta[n-1]
                    mu[n] = mu[n-1] + rhs*db
                    n += 1

            if ( 1.0/(kB*beta[n]) > Tmax ):
                res["converged"] = True
            if ( res["converged"] ):
                res["temperature"] = list( 1.0/(kB*beta[:n]) )
                res["mu"] = list( mu[:n] )
                res["singlet1"] = list( comp1[:n] )
                res["singlet2"] = list( comp2[:n] )
                return res

        res["temperature"] = list( 1.0/(kB*beta) )
        res["mu"] = list( mu )
        res["singlet1"] = list( comp1 )
        res["singlet2"] = list( comp2 )
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
