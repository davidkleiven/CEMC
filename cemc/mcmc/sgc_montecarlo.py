import montecarlo as mc
from cemc.mcmc.mc_observers import SGCObserver
import numpy as np
from ase.units import kB
import copy
from scipy import stats

class SGCMonteCarlo( mc.Montecarlo ):
    def __init__( self, atoms, temp, indeces=None, symbols=None, mpicomm=None, logfile="", plot_debug=False ):
        mc.Montecarlo.__init__( self, atoms, temp, indeces=indeces, mpicomm=mpicomm, logfile=logfile, plot_debug=plot_debug )
        if ( not symbols is None ):
            # Override the symbols function in the main class
            self.symbols = symbols
        self.averager = SGCObserver( self.atoms._calc, self, len(self.symbols)-1 )
        self.chem_pots = []
        self.chem_pot_names = []
        self.has_attached_avg = False
        self.name = "SGCMonteCarlo"
        self._chemical_potential = None
        self.chem_pot_in_ecis = False
        self.composition_correlation_time = np.zeros( len(self.symbols)-1 )

        has_attached_obs = False
        for obs in self.observers:
            if ( obs.name == "SGCObserver" ):
                has_attached_obs = True
                self.averager = obs
                break
        if ( not has_attached_obs ):
            self.attach( self.averager )

    def get_trial_move( self ):
        indx = np.random.randint( low=0, high=len(self.atoms) )
        old_symb = self.atoms[indx].symbol
        new_symb = old_symb
        while( new_symb == old_symb ):
            new_symb = self.symbols[np.random.randint(low=0,high=len(self.symbols))]
        system_changes = [(indx,old_symb,new_symb)]
        return system_changes

    def update_tracker( self, system_changes ):
        """
        Override the update of the atom tracker. The atom tracker is irrelevant in the semi grand canonical ensemble
        """
        pass

    def get_var_average_singlets( self ):
        """
        Returns the variance for the average singlets taking the correlation time into account
        """
        self.collect_averager_results()
        N = self.averager.counter
        singlets = self.averager.quantities["singlets"]/N
        singlets_sq = self.averager.quantities["singlets_sq"]/N
        var_n = ( singlets_sq - singlets**2 )
        #var_n = self.averager.quantities["singlet_diff"]/N

        if ( np.min(var_n) < -1E-5 ):
            msg = "The computed variances is {}".format(var_n)
            msg += "This is significantly smaller than zero and cannot be "
            msg += "attributed to numerical precission!"
            self.log( msg )

        nproc = 1
        if ( self.mpicomm is not None ):
            nproc = self.mpicomm.Get_size()

        if ( self.correlation_info is None or not self.correlation_info["correlation_time_found"] ):
            return var_n/(N*nproc)

        if ( not np.all(var_n>0.0) ):
            self.logger.warning( "Some variance where smaller than zero. (Probably due to numerical precission)" )
            self.log( "Variances: {}".format(var_n))
            var_n = np.abs(var_n)
        tau = self.correlation_info["correlation_time"]
        if ( tau < 1.0 ):
            tau = 1.0
        return 2.0*var_n*tau/(N*nproc)

    def has_converged_prec_mode( self, prec=0.01, confidence_level=0.05 ):
        """
        Checks that the averages have converged to the desired precission
        """
        energy_converged = super( SGCMonteCarlo, self ).has_converged_prec_mode( prec=prec, confidence_level=confidence_level )
        percentile = stats.norm.ppf(1.0-confidence_level)
        var_n = self.get_var_average_singlets()
        if ( self.mpicomm is not None ):
            var_n /= self.mpicomm.Get_size()
        singlet_converged = ( np.max(var_n) < (prec/percentile)**2 )
        #print ("{}-{}-{}".format(self.rank,singlet_converged,energy_converged))

        result = singlet_converged and energy_converged
        if ( self.mpicomm is not None ):
            send_buf = np.zeros(1,dtype=np.uint8)
            recv_buf = np.zeros(1,dtype=np.uint8)
            send_buf[0] = result
            self.mpicomm.Allreduce( send_buf, recv_buf )
            result = (recv_buf[0] == self.mpicomm.Get_size())
        return result

    def on_converged_log(self):
        """
        Log the convergence message
        """
        super(SGCMonteCarlo,self).on_converged_log()
        singlets = self.averager.singlets/self.averager.counter
        var_n = self.get_var_average_singlets()
        var_n = np.abs(var_n) # Just in case some variances should be negative
        self.log( "Thermal averaged singlet terms:" )
        for i in range( len(singlets) ):
            self.log( "{}: {} +- {}%".format(self.chem_pot_names[i],singlets[i],np.sqrt(var_n[i])/np.abs(singlets[i]) ) )

    def composition_reached_equillibrium(self, prev_composition, var_prev, confidence_level=0.05):
        """
        Returns True if the composition reached equillibrium
        """
        min_percentile = stats.norm.ppf(confidence_level)
        max_percentile = stats.norm.ppf(1.0-confidence_level)
        nproc = 1
        if ( self.mpicomm is not None ):
            nproc = self.mpicomm.Get_size()
        # Collect the result from the other processes
        # and average them into the values on the root node
        self.collect_averager_results()
        N = self.averager.counter
        singlets = self.averager.singlets/N
        singlets_sq = self.averager.quantities["singlets_sq"]/N
        var_n = self.get_var_average_singlets()

        if ( len(prev_composition) != len(singlets) ):
            # Prev composition is unknown so makes no sense
            # to check
            return False, singlets, var_n

        # Just in case variance should be smaller than zero. Should never
        # happen but can happen due to numerical precission
        var_n[var_n<0.0] = 0.0

        var_n /= nproc
        diff = singlets - prev_composition
        var_diff = var_n + var_prev
        if ( len(var_diff[var_diff>0.0]) == 0 ):
            return True, singlets, var_n
        z = np.max( np.abs(diff[var_diff>0.0])/np.sqrt(var_diff[var_diff>0.0]) )
        converged = False
        if ( z > min_percentile and z < max_percentile ):
            converged = True

        if ( self.mpicomm is not None ):
            # Broadcast the result to the other processors
            converged = self.mpicomm.bcast(converged,root=0)
            singlets = self.mpicomm.bcast(singlets,root=0)
            var_n = self.mpicomm.bcast(var_n,root=0)
        return converged, singlets, var_n




    def reset(self):
        """
        Reset the simulation object
        """
        super(SGCMonteCarlo,self).reset()
        self.averager.reset()

    @property
    def chemical_potential( self ):
        return self._chemical_potential

    @chemical_potential.setter
    def chemical_potential( self, chem_pot ):
        self._chemical_potential = chem_pot
        if ( self.chem_pot_in_ecis ):
            self.reset_eci_to_original( self.atoms._calc.eci)
        self.include_chemcical_potential_in_ecis( chem_pot, self.atoms._calc.eci )

    def include_chemcical_potential_in_ecis( self, chem_potential, eci ):
        """
        Including the chemical potentials in the ecis
        """
        self.chem_pots = []
        self.chem_pot_names = []
        keys = chem_potential.keys()
        keys.sort()
        for key in keys:
            self.chem_pots.append( chem_potential[key] )
            self.chem_pot_names.append(key)
            eci[key] -= chem_potential[key]
        self.atoms._calc.update_ecis( eci )
        self.chem_pot_in_ecis = True
        return eci

    def reset_eci_to_original( self, eci_with_chem_pot ):
        """
        Resets the ecis to their original value
        """
        for name,val in zip(self.chem_pot_names,self.chem_pots):
            eci_with_chem_pot[name] += val
        self.atoms._calc.update_ecis( eci_with_chem_pot )
        self.chem_pot_in_ecis = False
        return eci_with_chem_pot

    def reset_ecis( self ):
        return self.reset_eci_to_original( self.atoms.bc._calc.eci )

    def estimate_correlation_time_composition( self, window_length=1000, restart=False ):
        """
        #Estimate the corralation time for energy and composition
        """
        mc.Montecarlo.estimate_correlation_time( self, window_length=window_length, restart=restart )
        singlets = [[] for _ in range(len(self.symbols)-1)]
        for i in range(window_length):
            self.averager.reset()
            self._mc_step()
            singl = self.averger.singlets
            for i in range(len(singl)):
                singlets[i].append(singl[i])

        corr_times = []
        window_length_too_short = False
        for dset in singlets:
            mean = np.mean(dset)
            centered_dset = np.array(dset)-mean
            corr = np.correlate( centered_dset, centered_dset, mode="full" )
            corr = corr[int(len(corr)/2):]
            var = np.var(centered_dset)
            corr /= (var*window_length)
            if ( np.min(corr) > 0.5 ):
                window_length_too_short = True
                corr_times.append( window_length )
            else:
                indx = 0
                for i in range(len(corr)):
                    if ( corr[i] < 0.5 ):
                        indx = i
                        break
                rho = 2.0**(-1.0/indx)
                tau = -1.0/np.log(rho)
                corr_times.append(tau)

        self.composition_correlation_time = np.array(corr_times)
        if ( self.mpicomm is not None ):
            send_buf = np.zeros(1, dtype=np.uint8)
            recv_buf = np.zeros(1, dtype=np.uint8)
            send_buf[0] = window_length_too_short
            self.mpicomm.Allreduce( send_buf, recv_buf )
            window_length_too_short = ( recv_buf[0] >= 1 )

        if ( window_length_too_short ):
            msg = "The window length is too short to estimate the correlation time."
            msg += " Using the window length as correlation time."
            self.log( msg )

        # Collect the correlation times from all processes
        if ( self.mpicomm is not None ):
            recv_buf = np.zeros_like( self.composition_correlation_time )
            size = self.mpicomm.Get_size()
            self.mpicomm.Allreduce( self.composition_correlation_time, recv_buf )
            self.composition_correlation_time = recv_buf/size
        self.log( "Correlation time for the compositions:" )
        self.log( "{}".format(self.composition_correlation_time) )

    def runMC( self, mode="fixed", steps = 10, verbose = False, chem_potential=None, equil=True, equil_params=None, prec_confidence=0.05, prec=0.01 ):
        self.set_seeds()
        self.reset()
        if ( self.mpicomm is not None ):
            self.mpicomm.barrier()

        if ( chem_potential is None and self.chemical_potential is None ):
            ex_chem_pot = {
                "c1_1":-0.1,
                "c1_2":0.05
            }
            raise ValueError( "No chemicalpotentials given. Has to be dictionary of the form {}".format(ex_chem_pot) )

        if ( chem_potential is not None ):
            self.chemical_potential = chem_potential
        self.reset()

        # Include vibrations in the ECIS, does nothing if no vibration ECIs are set
        self.include_vib()

        if ( equil ):
            maxiter = 1000
            confidence_level = 0.05
            window_length = "auto"
            if ( equil_params is not None ):
                for key,value in equil_params.iteritems():
                    if ( key == "maxiter" ):
                        maxiter = value
                    elif ( key == "confidence_level" ):
                        confidence_level = value
                    elif ( key == "window_length" ):
                        window_length = value
            reached_equil = True
            res = self.estimate_correlation_time( restart=True )
            if ( not res["correlation_time_found"] ):
                res["correlation_time_found"] = True
                res["correlation_time"] = 1000
            self.distribute_correlation_time()
            try:
                self.equillibriate( window_length=window_length, confidence_level=confidence_level, maxiter=maxiter )
            except mc.DidNotReachEquillibriumError:
                reached_equil = False
            atleast_one_proc = self.atleast_one_reached_equillibrium(reached_equil)

            if ( not atleast_one_proc ):
                raise mc.DidNotReachEquillibriumError()

        self.reset()
        mc.Montecarlo.runMC( self, steps=steps, verbose=verbose, equil=False, mode=mode, prec_confidence=prec_confidence, prec=prec )

    def collect_averager_results(self):
        """
        If MPI is used, this function collects the results from the averager
        """
        if ( self.mpicomm is None ):
            return

        size = self.mpicomm.Get_size()
        all_res = self.mpicomm.gather( self.averager.quantities, root=0 )

        # Check that all processors have performed the same number of steps (which they should)
        same_number_of_steps = True
        msg = ""
        if ( self.rank == 0 ):
            for i in range(1,len(all_res)):
                if ( all_res[i]["counter"] != all_res[0]["counter"] ):
                    same_number_of_steps = False
                    msg = "Processor {} have performed a different number steps compared to 0.".format(i)
                    msg += "Number of stest {}: {}. Number of steps 0: {}".format( i, all_res[i]["counter"], all_res[0]["counter"])
                    break
        same_number_of_steps = self.mpicomm.bcast( same_number_of_steps, root=0 )

        if ( not same_number_of_steps ):
            raise RuntimeError( msg )

        par_works = True
        if ( self.rank == 0 ):
            par_works = self.parallelization_works( all_res )
        par_works = self.mpicomm.bcast( par_works, root=0 )
        if ( not par_works ):
            # This can happen either because the seed on all processors are the same
            # or because the results hav already been collected
            return
            msg = "It seems like exactly the same process is running on multiple processors!"
            raise RuntimeError( msg )

        # Average all the results from the all the processors
        if ( self.rank == 0 ):
            self.averager.quantities = all_res[0]
            for i in range(1,len(all_res)):
                for key,value in all_res[i].iteritems():
                    self.averager.quantities[key] += value/float(size)

                # Normalize by the number of processors
                #for key in self.averager.quantities.keys():
                #    self.averager.quantities[key] /= size

            """
            # Numerical pressicion issue?
            self.averager.quantities["singlet_diff"] = all_res[0]["singlets_sq"] - all_res[0]["singlets"]**2
            N = self.averager.counter
            for i in range(1,len(all_res)):
                self.averager.quantities["singlet_diff"] += all_res[i]["singlets_sq"] - all_res[i]["singlets"]**2
            self.averager.quantities["singlet_diff"] /= (size*N)
            """

        # Broadcast the averaged results
        self.averager.quantities = self.mpicomm.bcast( self.averager.quantities, root=0 )

    def get_thermodynamic( self, reset_ecis=True ):
        self.collect_averager_results()
        N = self.averager.counter
        quantities = {}
        singlets = self.averager.singlets/N
        singlets_sq = self.averager.quantities["singlets_sq"]/N
        #quantities["chem_pots"] = self.chem_pots
        quantities["energy"] = self.averager.energy/N
        for i in range( len(self.chem_pots) ):
            quantities["energy"] += self.chem_pots[i]*singlets[i]*len(self.atoms)

        quantities["heat_capacity"] = self.averager.energy_sq/N - (self.averager.energy/N)**2 + \
                                      np.sum( self.averager.singl_eng/N - (self.averager.energy/N)*singlets )
        quantities["heat_capacity"] /= (kB*self.T**2)
        quantities["temperature"] = self.T
        quantities["n_mc_steps"] = self.averager.counter
        # Add singlets and chemical potential to the dictionary
        for i in range(len(singlets)):
            quantities["singlet_{}".format(self.chem_pot_names[i])] = singlets[i]
            quantities["var_singlet_{}".format(self.chem_pot_names[i])] = singlets_sq[i]-singlets[i]**2
            quantities["mu_{}".format(self.chem_pot_names[i])] = self.chem_pots[i]
        if ( reset_ecis ):
            self.reset_eci_to_original( self.atoms._calc.eci )
        return quantities

    def parallelization_works( self, all_res ):
        """
        Checks that the entries in all_res are different.
        If not it seems like the same process is running on
        all the processors
        """
        if ( all_res is None ):
            return True

        ref_proc = all_res[-1] # Use the last processor as reference
        for i in range(0,len(all_res)-1):
            for key in ref_proc.keys():
                if ( key == "counter" ):
                    continue

                if ( isinstance( ref_proc[key],np.ndarray) ):
                    if ( not np.allclose( ref_proc[key], all_res[i][key] ) ):
                        return True
                else:
                    if ( ref_proc[key] != all_res[i][key] ):
                        return True
        return False
