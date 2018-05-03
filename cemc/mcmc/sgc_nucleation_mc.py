from nucleation_sampler import NucleationSampler, Mode
from cemc.mcmc import SGCMonteCarlo
from mc_observers import NetworkObserver

class SGCNucleation( SGCMonteCarlo ):
    def __init__( self, atoms, temp, **kwargs ):
        self.nuc_sampler = kwargs.pop("nucleation_sampler")
        kwargs["mpicomm"] = None
        self.network_name = kwargs.pop("network_name")
        self.network_element = kwargs.pop("network_element")
        chem_pot = kwargs.pop("chem_pot")
        super( SGCNucleation, self ).__init__( atoms, temp, **kwargs)
        self.chemical_potential = chem_pot

        self.network = NetworkObserver( calc=self.atoms._calc, cluster_name=self.network_name, element=self.network_element )
        self.attach( self.network )

    def accept( self, system_changes ):
        move_accepted = SGCMonteCarlo.accept( self, system_changes )
        return move_accepted and self.nuc_sampler.is_in_window(self.network)

    def get_trial_move(self):
        """
        Perform a trial move
        """
        if ( not self.nuc_sampler.is_in_window(self.network) ):
            raise RuntimeError( "System is outside the window before the trial move is performed!" )
        return SGCMonteCarlo.get_trial_move(self)

    def run( self, nsteps=10000 ):
        """
        Run samples in each window until a desired precission is found
        """
        if ( self.nuc_sampler.nucleation_mpicomm is not None ):
            self.nuc_sampler.nucleation_mpicomm.barrier()
        for i in range(self.nuc_sampler.n_windows):
            self.log( "Window {} of {}".format(i,self.nuc_sampler.n_windows) )
            self.nuc_sampler.current_window = i
            self.reset()
            self.nuc_sampler.bring_system_into_window(self.network)

            self.nuc_sampler.mode = Mode.equillibriate
            self.estimate_correlation_time()
            self.equillibriate()
            self.nuc_sampler.mode = Mode.sample_in_window

            current_step = 0
            while( current_step < nsteps ):
                current_step += 1
                self._mc_step()
                self.nuc_sampler.update_histogram(self)
                self.network.reset()

        if ( self.nuc_sampler.nucleation_mpicomm is not None ):
            self.nuc_sampler.nucleation_mpicomm.barrier()
