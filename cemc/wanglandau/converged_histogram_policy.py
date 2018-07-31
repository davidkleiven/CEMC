from cemc.wanglandau.settings import SimulationState

class ConvergedHistogramPolicy( object ):
    """
    This class (and its derived versions) determine the action taken
    when the histogram in the WL algorithm has converged
    """
    def __init__( self, wl_sim ):
        from cemc.wanglandau import WangLandau
        if ( not isinstance(wl_sim, WangLandau) ):
            raise TypeError( "wl_sim has to be of type WangLandau" )
        self.wl_sim = wl_sim

    def __call__( self ):
        """
        Executes the action taken. Defaut is no action and the simulation is
        converged
        """
        return SimulationState.CONVERGED

class LowerModificationFactor( ConvergedHistogramPolicy ):
    def __init__( self, wl_sim, m=2, fmin=1E-8 ):
        ConvergedHistogramPolicy.__init__(self,wl_sim)
        self.m = m
        self.fmin = fmin

    def __call__( self ):
        """
        Divide the modification factor by m reset the histogram
        """
        if ( self.wl_sim.f <= self.fmin ):
            return SimulationState.CONVERGED
        self.wl_sim.f /= self.m
        self.wl_sim.logger.info( "Resetting histogram" )
        self.wl_sim.histogram.histogram[:] = 0.0
        self.wl_sim.logger.info( "Lowering the modification factor. New f: {}".format(self.wl_sim.f))
        return SimulationState.CONTINUE

class DoNothingContinueUntilModFactorIsConverged( ConvergedHistogramPolicy ):
    def __init_( self, wl_sim, fmin=1E-8 ):
        ConvergedHistogramPolicy.__init__( self, wl_sim )
        self.fmin = fmin

    def __call__( self ):
        """
        Continue if f is to large
        """
        if ( self.wl_sim.f > self.fmin ):
            return SimulationState.CONTINUE
        return SimulationState.CONVERGED
