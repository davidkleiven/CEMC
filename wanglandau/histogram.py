import numpy as np
from scipy import interpolate
import wltools
import sqlite3 as sq

class Histogram( object ):
    """
    Class for tracking the WL histtogram and related quantities
    """
    def __init__( self, Nbins, Emin, Emax, logger ):
        self.Nbins = Nbins
        self.Emin = Emin
        self.Emax = Emax
        self.histogram = np.zeros(self.Nbins, dtype=np.int32)
        self.logdos = np.zeros( self.Nbins )
        self.growth_variance = np.zeros( self.Nbins )
        self.logger = logger

    def get_energy( self, bin ):
        """
        Returns the energy corresponding to one bin
        """
        return self.Emin + (self.Emax-self.Emin )*indx/self.Nbins

    def get_bin( self, energy ):
        """
        Returns the bin corresponding to one energy
        """
        return int( (energy-self.Emin)*self.Nbins/(self.Emax-self.Emin) )

    def update( self, selected_bin, mod_factor ):
        """
        Updates all quantities
        """
        self.histogram[selected_bin] += 1
        self.logdos[selected_bin] += mod_factor
        self.growth_variance += self.Nbins**(-2)
        self.growth_variance[selected_bin] += (1.0 - 2.0/self.Nbins)

    def update_range( self ):
        """
        Updates the range of the histogram to better fit the required energies
        """
        upper = self.Nbins
        for i in range(len(self.histogram)-1,0,-1):
            if ( self.histogram[i] > 0 ):
                upper = i
                break
        lower = 0
        for i in range(len(self.histogram)):
            if ( self.histogram[i] > 0 ):
                lower = i
                break

        Emin = self.get_energy(lower)
        Emax = self.get_energy(upper)
        if ( Emax < self.largest_energy_ever ):
            Emax = self.largest_energy_ever
        if ( Emin > self.smallest_energy_ever ):
            Emin = self.smallest_energy_ever
        if ( Emin != self.Emin or Emax != self.Emax ):
            self.redistribute_hist(Emin,Emax)

    def redistribute_hist( self, Emin, Emax ):
        """
        Redistributes the histogram to better fit the new energies
        """
        if ( Emin > Emax ):
            Emin = Emax-10.0
        eps = 1E-8
        Emax += eps
        old_E = np.linspace( self.Emin, self.Emax, self.Nbins )
        new_E = np.linspace( Emin, Emax, self.Nbins )
        interp_hist = interpolate.interp1d( old_E, self.histogram, bounds_error=False, fill_value=0 )
        new_hist = interp_hist(new_E)
        interp_logdos = interpolate.interp1d( old_E, self.logdos, bounds_error=False, fill_value=0 )
        new_logdos = interp_logdos(new_E)
        interp_var = interpolate.interp1d( old_E, self.cummulated_variance, bounds_error=False, fill_value="extrapolate" )
        self.cummulated_variance = interp_var( new_E )

        # Scale
        if ( np.sum(new_hist) > 0 ):
            new_hist *= np.sum(self.histogram)/np.sum(new_hist)
        if ( np.sum(new_logdos) > 0 ):
            new_logdos *= np.sum(self.logdos)/np.sum(new_logdos)
        self.histogram = np.floor(new_hist).astype(np.int32)
        self.logdos = new_logdos
        for i in range(len(self.histogram)):
            if ( self.histogram[i] == 0 ):
                # Set the DOS to 1 if the histogram indicates that it has never been visited
                # This just an artifact of the interpolation and setting it low will make
                # sure that these parts of the space gets explored
                self.logdos[i] = 0.0
        self.Emin = Emin
        self.Emax = Emax

    def get_growth_fluctuation( self ):
        """
        Returns the fluctuation of the growth term
        """
        N = np.sum(self.histogram)
        if ( N <= 1 ):
            return None
        std = np.sqrt( self.growth_variance/N )
        return std

    def std_check( self ):
        """
        Check that all bins (with a known structure is larger than 1000 times the standard deviation)
        """
        factor = 1000.0
        if ( np.sum(self.histogram) < 20 ):
            return False

        growth_fluct = self.get_growth_fluctuation()
        converged = True
        self.tot_number = 0
        self.number_of_converged = 0
        for i in range(len(self.histogram)):
            if ( self.histogram[i] == 0 ):
                continue
            if ( self.histogram[i] <= factor*growth_fluct[i] ):
                converged = False
            else:
                self.number_of_converged += 1
            self.tot_number += 1
        return converged

    def is_flat( self, criteria ):
        """
        Checks whether the histogram satisfies the flatness criteria
        """
        if ( np.sum(self.histogram) < 100 ):
            return False

        mask = np.zeros(len(self.histogram),dtype=np.int8)
        mask[:] = True
        for i in range(len(self.histogram)):
            if ( self.histogram[i] == 0 and self.structures[i] is None ):
                mask[i] = False
        mean = np.mean( self.histogram[mask] )
        self.tot_number = np.sum(mask)
        self.number_of_converged = np.sum(self.histogram[mask]>self.flatness_criteria*mean)
        return np.min(self.histogram[mask]) > self.flatness_criteria*mean

    def log_progress( self ):
        """
        Logs the progress of the run
        """
        current_time = time.localtime()
        timestr = time.strftime( "%H:%M:%S", current_time)
        self.logger.info( "%s %d of %d bins (with known structures) has converged"%(timestr,number_of_converged,tot_number))

    def clear( self ):
        """
        Clears the histogram
        """
        self.histogram[:] = 0
        self.logdos[:] = 0.0
        self.growth_variance[:] = 0.0

    def load( self, db_name, uid ):
        """
        Loads results from a previuos run from the database
        """
        conn = sq.connect( db_name )
        cur = conn.cursor()
        sql = "select histogram,logdos,growth_variance,Emin,Emax from simulations where uid=?"
        cur.execute( sql, (uid,) )
        entries = cur.fetchone()
        conn.close()

        try:
            self.histogram = wltools.convert_array( entries[0] )
            self.logdos = wltools.convert_array( entries[1] )
            self.growth_variance = wltools.convert_array( entries[2] )
        except Exception as exc:
            self.logger.warning( "The following exception occured during load data from the database")
            self.logger.warning( str(exc) )
            self.logger.warning( "The sampling will be started from scratch" )
        self.Emin = float(entries[3])
        self.Emax = float(entries[4])
        self.Nbins = len(self.histogram)

    def save( self, db_name, uid ):
        """
        Stores all arrays to the database
        """
        conn = sq.connect( db_name )
        cur = conn.cursor()
        E = np.linspace( self.Emin, self.Emax, self.Nbins )
        cur.execute( "update simulations set energy=? WHERE uid=?", (wltools.adapt_array(E),uid)  )
        cur.execute( "update simulations set logdos=? WHERE uid=?", (wltools.adapt_array(self.logdos),uid)  )
        cur.execute( "update simulations set histogram=? WHERE uid=?", (wltools.adapt_array(self.histogram),uid) )
        cur.execute( "update simulations set growth_variance=? WHERE uid=?", (wltools.adapt_array(self.growth_variance),uid))
        cur.execute( "update simulations set Emin=?, Emax=? where uid=?", (self.Emin,self.Emax,uid) )
        conn.commit()
        conn.close()
