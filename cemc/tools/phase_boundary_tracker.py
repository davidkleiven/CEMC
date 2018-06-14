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
from matplotlib import pyplot as plt
plt.switch_backend("Agg") # With this backend one does not need a screen (useful for clusters)
#import Image

comm = MPI.COMM_WORLD

class PhaseChangedOnFirstIterationError(Exception):
    def __init__( self, message ):
        super( PhaseChangedOnFirstIterationError,self).__init__(message)

class PhaseBoundaryTracker(object):
    def __init__( self, ground_states, logfile="", chemical_potential=None, \
    conf_same_phase=0.45, conf_substep_conv=0.05, max_singlet_change=0.05, backupfile="backup_phase_track.h5" ):
        """
        Class for tracking the phase boundary between two phases
        NOTE: Has only been confirmed to work for binary systems!

        :params gs: List of dictionaries containing the following fields
                    * *bc* BulkCrystal object,
                    * *cf* Correlation function of the state of the atoms object in BulkCrystal
                    * *eci* Effective Cluster Interactions
        :param logfile: Filename where log messages should be written
        :param conf_same_phase: Confidence level used for hypothesis testing when checking if the two systems
                                have ended up in the same phase. Has to be in the range [0.0,0.5)
        :param conf_substep_conv: Confidence level used for hypethesis testing when checking if the stepsize
                            needs to be refined. Has to be in the range[0.0,0.5)
        :param max_singlet_change: The maximum amount the singlet terms are allowed to change on each step
        """
        self.ground_states = ground_states
        for gs in self.ground_states:
            self.check_gs_argument(gs)

        self._conf_same_phase = conf_same_phase
        self._conf_substep_conv = conf_substep_conv
        self.max_singlet_change = max_singlet_change
        self.backupfile = backupfile

        if ( os.path.exists(self.backupfile) ):
            os.remove(self.backupfile)

        #if ( chemical_potential is None ):
        #    self.chemical_potential = {mu_name:0.0}
        #else:
        #    self.chemical_potential=chemical_potential

        self.singlet_names = [name for name in self.ground_states[0]["cf"].keys() if name.startswith("c1")]

        # Set the calculators of the atoms objects
        calcs = []
        self.mfa = []
        #for gs in self.ground_states:
        #    calcs.append( ClusterExpansion( gs["bc"], cluster_name_eci=gs["eci"], init_cf=gs["cf"] ) )
        #    gs["bc"].atoms.set_calculator(calcs[-1])
        #    self.mfa.append( MeanFieldApprox(gs["bc"]) )

        self.current_backup_indx = 0

        self.logger = logging.getLogger( "PhaseBoundaryTracker" )
        self.logger.setLevel( logging.INFO )

        if logfile == "":
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

    def check_gs_argument(self, gs):
        """
        Check that the gs arguments contain the correct fields

        :param gs: Ground state structure
        """
        required_fields = ["bc","cf","eci"]
        keys = gs.keys()
        for key in keys:
            if ( key not in required_fields ):
                raise ValueError( "The GS argument has to contain {} keys. Given {}".format(required_fields,keys) )

    def get_gs_energies(self):
        """
        Return the Ground State Energies
        """
        energy = []
        for gs in self.ground_states:
            gs_energy = 0.0
            for key in gs["eci"].keys():
                gs_energy += gs["eci"][key]*gs["cf"][key]
            energy.append(len(gs["bc"].atoms)*gs_energy)
        return energy

    def get_zero_temperature_mu_boundary( self ):
        """
        Computes the chemical potential at which the two phases coexists
        at zero kelvin
        """
        N = len(self.ground_states)-1
        B = np.zeros((N,N))
        energy_vector = np.zeros(N)
        gs_energies = self.get_gs_energies()
        for i in range(N):
            for j in range(N):
                ref_singlet = self.ground_states[0]["cf"][self.singlet_names[j]]
                singlet = self.ground_states[i+1]["cf"][self.singlet_names[j]]
                B[i,j] = (ref_singlet-singlet)
            #E_ref = self.ground_states[0]["bc"].atoms.get_potential_energy()/len(self.ground_states[0]["bc"].atoms)
            #E = self.ground_states[i+1]["bc"].atoms.get_potential_energy()/len(self.ground_states[i+1]["bc"].atoms)
            E_ref = gs_energies[0]/len(self.ground_states[0]["bc"].atoms)
            E = gs_energies[i+1]/len(self.ground_states[i+1]["bc"].atoms)
            energy_vector[i] = E_ref-E

        mu_boundary = np.linalg.solve(B, energy_vector)
        return mu_boundary

        """
        # Assuming that the chemical potential at this point is not included into the ECIs
        E1 = self.gs1["bc"].atoms.get_potential_energy()
        E2 = self.gs2["bc"].atoms.get_potential_energy()
        x1 = self.gs1["cf"][self.mu_name]
        x2 = self.gs2["cf"][self.mu_name]
        mu_boundary = (E2-E1)/(x2-x1)
        return mu_boundary/len( self.gs1["bc"].atoms )
        """

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

        :param data: Dictionary of data to be backed up
        :param dsetname: Basename for all datasets in the h5 file
        """
        rank = comm.Get_rank()
        if ( rank == 0 ):
            with h5.File( self.backupfile, 'a' ) as f:
                grp = f.create_group(dsetname+"{}".format(self.current_backup_indx))
                for key,value in data.iteritems():
                    if ( value is None ):
                        continue
                    if ( key == "images" ):
                        for img_num, img in enumerate(value):
                            if img is None:
                                continue
                            #img = img.T
                            dset = grp.create_dataset( "img_{}".format(img_num), data=img )
                            dset.attrs['CLASS'] = "IMAGE"
                            dset.attrs['IMAGE_VERSION'] = '1.2'
                            dset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_INDEXED'
                            dset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)
                    else:
                        grp.create_dataset( key, data=value )
            self.current_backup_indx += 1
        comm.barrier()

    def predict_composition( self, comp, temperatures, target_temp, target_comp=0.0 ):
        """
        Performs a prediction of the next composition value based on history

        :param comp: History of compositions
        :param temperatures: History of temperatures
        :param target_temp: Temperature where the composition should be predicted
        :param target_comp: Not used
        """
        if ( len(comp) <= 2 ):
            return comp[-1], None
        if ( len(comp) == 3 ):
            k = 2
        else:
            k = 3
        x = np.arange(0,len(temperatures))[::-1]
        weights = np.exp(-2.0*x/len(temperatures) )

        # Ad hoc smoothing parameter
        # This choice leads to the deviation from the last point being
        # maximum 0.05
        smoothing = 0.05*np.sum(weights)


        # Weight the data sich that the last point is more important than
        # the first.
        # Impact of the first point is exp(-2) relative to the impact of the
        # last point
        spl = UnivariateSpline( temperatures, comp, k=k, w=weights, s=smoothing )
        predicted_comp = spl(target_temp)

        rgbimage = None
        rank = comm.Get_rank()
        if ( rank == 0 ):
            # Create a plot of how the spline performs
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot( temperatures, comp, marker='^', color="black", label="History" )
            x = np.linspace( np.min(temperatures), target_temp+40, 100 )
            pred = spl(x)
            ax.plot( x, pred, "--", label="Spline" )
            ax.plot( [target_temp], [target_comp], 'x', label="Computed" )
            ax.set_ylabel( "Singlets" )
            ax.set_xlabel( "Temperature (K)" )
            ax.legend()
            rgbimage = fig2rgb(fig)
            plt.close("all")
        rgbimage = comm.bcast( rgbimage, root=0 )
        return predicted_comp, rgbimage

    def is_equal( self, x1, x2, std1, std2, confidence_level=0.05, stdtol=1E-6, eps_fallback=0.05 ):
        """
        Check if two numbers are equal provided that their standard deviations
        are known

        :param x1: First value
        :param x2: Second value
        :param std1: Standard deviation of the first value
        :param std2: Standard deviation of the second value
        :param confidence_level: Confidence level for hypothesis testing
        :param stdtol: If the standard deviation of the difference between x1 and x2
            is smaller than this value. It will not perform hypothesis testing.
        :param eps_fallback: If not hypothesis returns abs(x1-x1) < eps_fallback
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

    def argsort_states_based_on_singlets(self, thermo):
        """
        Returns a sorted list
        """
        n_singlets = len(self.singlet_names)
        n_states = len(self.ground_states)
        sorted_indices = []
        for name in self.singlet_names:
            singlet_values = []
            indices = range(len(thermo))
            for entry in thermo:
                singlet_values.append(entry[self.get_singlet_name(name)])
            srt_indx = np.argsort(singlet_values)
            sorted_indices.append([indices[indx] for indx in srt_indx])
        return np.array(sorted_indices)

    def composition_first_larger_than_second( self, thermo ):
        """
        Check if the first arguments is larger than the second

        :param x1: Create an array
        :param x2: Second value
        """
        N = len(self.ground_states)-1
        return x1 > x2

    def system_changed_phase( self, prev_comp, comp ):
        """
        Check if composition changes too much from one step to another

        :param prev_comp: Composition on the previous step
        :param comp: Composition on the current step
        """
        return np.abs( prev_comp-comp ) > self.max_singlet_change

    def reset_symbols( self, calc, old_symbs ):
        """
        Resetting symbol and print a log message of the new elements

        :param calc: CE calculator object
        :param old_symbs: Symbols of old state
        """
        calc.set_symbols( old_symbs )
        cf = calc.get_cf()
        singlets = {}
        for key,value in cf.iteritems():
            if ( key.startswith("c1") ):
                singlets[key] = value
        self.log( "Singlets in system: {}".format(singlets) )


    def get_mu_dict(self, mu_vec):
        """
        Returns a dictionary of mu based on the values in a numby array
        """
        mu_dict = {}
        for i,name in enumerate(self.singlet_names):
            mu_dict[name]= mu_vec[i]
        return mu_dict

    def get_singlet_name(self, orig_name):
        """
        Returns the singlet name as stored in the thermo-dictionary
        """
        return "singlet_{}".format(orig_name)


    def get_rhs(self, thermo, mu_array, beta):
        """
        Computes the right hand side of the phase boundary equation
        """
        N = len(self.ground_states)-1
        A = np.zeros((N,N))
        energy_vector = np.zeros(N)
        print(N)
        for i in range(N):
            for j in range(N):
                ref_singlet = thermo[0][self.get_singlet_name(self.singlet_names[j])]
                singlet = thermo[i+1][self.get_singlet_name(self.singlet_names[j])]
                A[i,j] = ref_singlet-singlet
            ref_energy = thermo[0]["energy"]/len(self.ground_states[0]["bc"].atoms)
            E = thermo[i+1]["energy"]/len(self.ground_states[i+1]["bc"].atoms)
            energy_vector[i] = ref_energy-E
        invA = np.linalg.inv(A)
        rhs = invA.dot(energy_vector)/beta - mu_array/beta
        return rhs

    def get_singlet_array(self, thermo):
        """
        Return array of the singlet terms
        """
        singlets = []
        for entry in thermo:
            singlets.append([entry[self.get_singlet_name(name)] for name in self.singlet_names])
        return singlets

    def get_singlet_evolution(self, singlet_history, phase_indx, singlet_indx ):
        """
        Return the history of one particular singlet term in one phase
        """
        history= np.zeros(len(singlet_history))
        for i,entry in enumerate(singlet_history):
            history[i] = entry[phase_indx][singlet_indx]
        return history

    def separation_line_adaptive_euler( self, T0=100, min_step=1, stepsize=100, mc_args={}, symbols=[] ):
        """
        Solve the differential equation using adaptive euler

        :param T0: Initial temperature
        :param min_step: Minimum step size in kelvin
        :param stepsize: Initial stepsize in kelving
        :param mc_args: Dictionary of arguments for the MC samplers.
            See :py:meth:`cemc.mcmc.SGCMonteCarlo.runMC`
        """
        mu_prev = self.get_zero_temperature_mu_boundary()
        mu = np.zeros_like(mu_prev)
        if ( comm.Get_size() > 1 ):
            mpicomm = comm
        else:
            mpicomm = None

        calcs = []
        old_symbols = []
        sgc_obj = []
        for gs in self.ground_states:
            calcs.append(CE(gs["bc"], gs["eci"], initial_cf=gs["cf"]))
            gs["bc"].atoms.set_calculator(calcs[-1])
            symbs = [atom.symbol for atom in gs["bc"].atoms]
            old_symbols.append(symbs)
            sgc_obj.append(SGCMonteCarlo(gs["bc"].atoms, T0, symbols=symbols, mpicomm=mpicomm))

        for calc in calcs:
            print(calc.get_cf())

        #calc1 = CE( self.gs1["bc"], self.gs1["eci"], initial_cf=self.gs1["cf"] )
        #calc2 = CE( self.gs2["bc"], self.gs2["eci"], initial_cf=self.gs2["cf"] )
        #self.gs1["bc"].atoms.set_calculator(calc1)
        #self.gs2["bc"].atoms.set_calculator(calc2)
        reference_logical_phase_check = False
        orig_stepsize = stepsize

        if ( "equil" in mc_args.keys() ):
            if ( not mc_args["equil"] ):
                self.log( "This scheme requires that the system can equillibriate. Changing to equil=True" )

        mc_args["equil"] = True

        singlets = []
        mu_array = []
        temp_array = []
        dT = stepsize
        Tprev = T0
        target_comp = None
        target_temp = None
        rhs_prev = np.zeros_like(mu)
        is_first = True
        mu[:] = mu_prev
        n_steps_required_to_reach_temp = 1
        substep_count = 1
        update_symbols = [True for _ in range(len(self.ground_states))]
        #update_symbols2 = True
        T = Tprev
        #singl_name = "singlet_{}".format(self.mu_name)
        ref_compare = False
        iteration = 1
        eps = 0.01

        #symbs1_old = [atom.symbol for atom in self.gs1["bc"].atoms]
        #symbs2_old = [atom.symbol for atom in self.gs2["bc"].atoms]
        while( stepsize > min_step ):
            self.log( "Current temperature {}K. Current chemical_potential: {} eV/atom".format(int(T),mu) )
            self.chemical_potential = self.get_mu_dict(mu)
            mc_args["chem_potential"] = self.chemical_potential
            Tnext = T+dT

            beta = 1.0/(kB*T)

            #sgc1.T = T
            #sgc2.T = T
            comm.barrier()
            thermo = []
            for i,sgc in enumerate(sgc_obj):
                self.log("Running MC for system {}".format(i))
                sgc.T = T
                sgc.runMC(**mc_args)
                thermo.append(sgc.get_thermodynamic())

            #sgc1.runMC( **mc_args )
            #thermo1 = sgc1.get_thermodynamic()
            #comm.barrier()
            #sgc2.runMC( **mc_args )
            #thermo2 = sgc2.get_thermodynamic()

            #x1 = thermo1[singl_name]
            #x2 = thermo2[singl_name]
            #E1 = thermo1["energy"]
            #E2 = thermo2["energy"]
            #rhs = (E2-E1)/( beta*(x2-x1)*len(sgc1.atoms) ) - mu/beta
            rhs = self.get_rhs(thermo, mu, beta)

            if ( is_first ):
                rhs_prev = rhs
                is_first = False
                beta_prev = 1.0/(kB*T)
                mu_prev[:] = mu
                Tprev = T
                singlets.append(self.get_singlet_array(thermo))
                ref_singlet = singlets[-1]
                #comp1.append(x1)
                #comp2.append(x2)
                temp_array.append(T)
                mu_array.append(mu)
                T += dT
                beta_next = 1.0/(kB*T)
                dbeta = beta_next-beta_prev
                mu += rhs*dbeta
                #ref_compare = self.composition_first_larger_than_second( x1, x2 )
                ref_order = self.argsort_states_based_on_singlets(thermo)
                continue

            """
            x1 = thermo1[singl_name]
            x2 = thermo2[singl_name]

            var_name = "var_singlet_{}".format(self.mu_name)
            std1 = np.sqrt( thermo1[var_name] )
            std2 = np.sqrt( thermo2[var_name] )
            eps = 0.01

            # Check if the two phases ended up in the same phase
            x1_equal_to_x2 = self.is_equal( x1, x2, std1, std2, confidence_level=0.45 )

            compositions_swapped = self.composition_first_larger_than_second(x1,x2) != ref_compare
            """
            sorted_phases = self.argsort_states_based_on_singlets(thermo)
            compositions_swapped = np.any(sorted_phases != ref_order)

            # Check if one of the systems changed phase
            #if ( len(comp1) == 0 or len(comp2) == 0 ):
            #    ref1 = x1
            #    ref2 = x2
            singlet_array = self.get_singlet_array(thermo)
            ref_values = []
            images = []

            if len(singlets) == 0:
                assert False # Should never be the case
            else:
                for i in range(len(self.ground_states)):
                    ref_val_in_state = []
                    for j in range(len(self.singlet_names)):
                        history = self.get_singlet_evolution(singlets, i, j)
                        ref, img = self.predict_composition( history, temp_array, T, target_comp=singlet_array[i][j])
                        ref_val_in_state.append(ref)
                        images.append(img)
                        self.log( "Temperatures used for prediction:")
                        self.log( "{}".format(temp_array) )
                        self.log( "Compositions system {}, singlet {}:".format(i,j) )
                        self.log( "{}".format(history) )
                        self.log( "Predicted composition for T={}K: {}".format(T,ref) )
                    ref_values.append(ref_val_in_state)
                """
                ref1, img1 = self.predict_composition( comp1, temp_array, T, target_comp=x1 )
                ref2, img2 = self.predict_composition( comp2, temp_array, T, target_comp=x2 )
                self.log( "Temperatures used for prediction:")
                self.log( "{}".format(temp_array) )
                self.log( "Compositions syst1:" )
                self.log( "{}".format(comp1) )
                self.log( "Predicted composition for T={}K: {}".format(T,ref1))
                self.log( "Compositions syst2:" )
                self.log( "{}".format(comp2) )
                self.log( "Predicted composition for T={}K: {}".format(T,ref2) )
                """

            one_system_changed_phase = False
            for i in range(len(singlet_array)):
                for j in range(len(singlet_array[0])):
                    if self.system_changed_phase(singlet_array[i][j], ref_values[i][j]):
                        one_system_changed_phase = True
                        break
            #one_system_changed_phase = self.system_changed_phase(x1,ref1) or self.system_changed_phase(x2,ref2)
            if ( compositions_swapped or one_system_changed_phase):
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
                mu[:] = mu_prev
                rhs[:] = rhs_prev

                for calc, symbs in zip(calcs, old_symbols):
                    self.reset_symbols(calc, symbs)
                #self.reset_symbols( calc1, symbs1_old )
                #self.reset_symbols( calc2, symbs2_old )

                self.log( "New step size: {}K".format(dT) )
                self.log( "New trial step size (used when computing the first target temperature after convergence): {}K".format(stepsize))
            else:
                #target_comp1 = ref1
                #target_comp2 = ref2

                # Compare with the reference composition
                #x1_is_equal = self.is_equal( target_comp1, x1, std1, std1, stdtol=1E-6, eps_fallback=eps )
                #x2_is_equal = self.is_equal( target_comp2, x2, std2, std2, stdtol=1E-6, eps_fallback=eps )
                all_equal = True
                for i in range(len(singlet_array)):
                    for j in range(len(singlet_array[0])):
                        if abs(singlet_array[i][j] - ref_values[i][j]) > eps:
                            all_equal = False

                #if ( x1_is_equal and x2_is_equal ):
                if all_equal:
                    self.log("============================================================")
                    self.log("== Converged. Proceeding to the next temperature interval ==")
                    self.log("============================================================")
                    # Converged
                    stepsize = orig_stepsize
                    dT = stepsize
                    Tprev = T
                    rhs_prev[:] = rhs
                    mu_prev[:] = mu
                    singlets.append(singlet_array)
                    #comp1.append(x1)
                    #comp2.append(x2)
                    temp_array.append(T)
                    mu_array.append(mu)
                    backupdata = {
                        "singlets":singlets,
                        "temperature":temp_array,
                        "mu":mu_array,
                        "images":images,
                    }
                    self.backup( backupdata, dsetname="iter")
                else:
                    # Did not converge reset and decrease the stepsize
                    T = Tprev
                    dT /= 2.0
                    mu[:] = mu_prev
                    rhs[:] = rhs_prev

                    # Update the target compositions to the new ones
                    self.log( "Did not converge. Updating target compositions. Refining stepsize. New stepsize: {}K".format(dT) )
                    self.log( "Resetting system" )

                    # Reset the system to pure phases
                    for calc, symbs in zip(calcs, old_symbols):
                        self.reset_symbols(calc, symbs)
                    #self.reset_symbols( calc1, symbs1_old )
                    #self.reset_symbols( calc2, symbs2_old )

            beta_prev = 1.0/(kB*T)
            T += dT
            beta_next = 1.0/(kB*T)
            dbeta = beta_next - beta_prev
            mu += rhs*dbeta

            # Append the last step to the array
            if ( stepsize <= min_step ):
                temp_array.append(T)
                mu_array.append(mu)
                singlets.append(singlet_array)
                #comp1.append(x1)
                #comp2.append(x2)

        res = {}
        res["temperature"] = temp_array
        res["mu"] = mu_array
        res["singlet"] = singlets
        res["msg"] = "Not able to make progress with the smalles stepsize {}K".format(min_step)
        return res

    def mean_field_separation_line( self, Tmax, nsteps=10 ):
        """
        Computes the separation line in the mean field approximation

        :param Tmax: Maximum temperature
        :param nsteps: Number of steps between T=1K and Tmax
        """
        mu0 = self.get_zero_temperature_mu_boundary()
        Tmin = 1.0
        beta = np.linspace(1.0/(kB*Tmin), 1.0/(kB*Tmax), nsteps )
        mu = np.zeros((nsteps,len(mu0)))
        mu[0,:] = mu0
        conc1 = np.zeros( nsteps )
        conc2 = np.zeros(nsteps)
        for n in range(1,nsteps):
            chem_pot = self.mu_dict(mu[n-1,:])
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
        Construction of a common tangent point (NOT USED)
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

        :param Tmin: Minimum temperature
        :param Tmax: Maximum temperature
        :param ntemps: Number of temperatures between Tmin and Tmax
        """
        betas = np.linspace( 1.0/(kB*Tmin), 1.0/(kB*Tmax), ntemps )
        mu0 = self.get_zero_temperature_mu_boundary()
        db = betas[1]-betas[0]
        delta_singlets = self.gs1["cf"][self.mu_name] - self.gs2["cf"][self.mu_name] # Constant in MFA

def fig2rgb( fig ):
    """
    Convert matplotlib figure instance to a png
    """
    fig.canvas.draw()

    # Get RGB values
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = (h,w,3)
    greyscale = 0.2989 * buf[:,:,0] + 0.5870 * buf[:,:,1] + 0.1140 * buf[:,:,2]
    greyscale = greyscale.astype(np.uint8)
    return greyscale
