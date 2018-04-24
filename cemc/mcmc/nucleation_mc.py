from cemc.mcmc import SGCMonteCarlo

class NucleationMC( SGCMonteCarlo ):
    def __init__( self, **kwargs ):
        self.size_window_width = kwargs.pop("size_window_width")
        super(NucleationMC,self).__init__(**kwargs)
        
