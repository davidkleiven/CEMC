from cemc.wanglandau.settings import SimulationState
from cemc.wanglandau import converged_histogram_policy as chp

class ModificationFactorUpdater( object ):
    def __init__( self, wl_sim ):
        from cemc.wanglandau import WangLandau
        if ( not isinstance(wl_sim, WangLandau) ):
            raise TypeError( "wl_sim has to be of type WangLandaus" )
        self.wl_sim = wl_sim

    def update( self ):
        """
        Updates the modification factor
        """
        return SimulationState.CONTINUE


class InverseTimeScheme( ModificationFactorUpdater ):
    """
    This class implements the modification scheme presented in

    Fast algorithm to calculate density of states, R.E. Belardinelli and V.D. Pereyra
    """

    def __init__( self, wl_sim, fmin=1E-6 ):
        ModificationFactorUpdater.__init__( self, wl_sim )
        self.inverse_time_scheme_active = False
        self.fmin = fmin

    def estimate_number_of_MC_steps_left( self ):
        """
        Estimates the number of MC left until the simulatinos has converged
        """
        return self.wl_sim.Nbins/self.fmin - self.wl_sim.iter

    def update( self ):
        """
        Updates the modification factor
        """
        if ( self.wl_sim.f <= self.fmin ):
            return SimulationState.CONVERGED

        time = float(self.wl_sim.iter)/self.wl_sim.Nbins
        if ( 1.0/time > 1.0 ):
            return SimulationState.CONTINUE

        if ( self.wl_sim.f < 1.0/time and not self.inverse_time_scheme_active ):
            self.wl_sim.logger.info( "Changing to inverse time scheme." )
            self.wl_sim.logger.info( "Changing converged histogram policy to \"Do Nothing\"")
            self.wl_sim.on_converged_hist = chp.DoNothingContinueUntilModFactorIsConverged( self.wl_sim, fmin=self.wl_sim.fmin )
            self.wl_sim.logger.info( "Number of MC steps until converged: {}".format(self.estimate_number_of_MC_steps_left()) )
            self.inverse_time_scheme_active = True

        if ( self.inverse_time_scheme_active ):
            self.wl_sim.f = 1.0/time
        return SimulationState.CONTINUE
