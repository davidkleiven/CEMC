
import numpy as np
import pickle as pkl
import ase.units as units
from matplotlib import pyplot as plt
from scipy import interpolate
import copy
import json
import sqlite3 as sq
import io
import wltools
from ase.db import connect
from ase.visualize import view
import mod_factor_updater as mfu
import converged_histogram_policy as chp
from settings import SimulationState
import logging
import time
from histogram import Histogram

class WangLandauSGC( object ):
    def __init__( self, db_name, db_id, site_types=None, site_elements=None, Nbins=100, initial_f=2.71,
    flatness_criteria=0.8, fmin=1E-6, Emin=0.0, Emax=1.0, conv_check="flathist", scheme="fixed_f",
    logfile="default.log", ensemble="canonical" ):
        """
        Class for running Wang Landau Simulations in the Semi Grand Cannonical Ensemble or Canonical

        Parameters
        -----------
        db_name - Name of database to store records
        db_id   - ID of the run to perform
        site_types - 1D array describing which site type each atom position belongs to. If None all atoms positions are assumed
                     to be of the same type and it is set to an array of [0,0,0,0 ..., 0,0]
        site_elements - List of lists of which elements are allowed on the different site types
                        [["Al","Mg"], ["Si"]] means for instance that on site type 0 only Al and Mg atoms are allowed,
                        while on site type 1 only Si atoms are allowed.
        Nbins - Number of bins in the histogram and DOS
        initial_f - Modification factor. See scheme
        flatness_criteria - Histogram is considered to be flat if min(histogram) > flatness_criteria*mean(histogram)
        fmin - Minimum modification factor (only relevant if scheme is lower_f)
        Emin - Initial guess for the minimum energy in the spectrum
        Emax - Initial guess for the maximum energy in the spectrum
        conv_check - How to determine convergence.
                     flathist - determined by flatness of the histogram (see flatness_criteria)
                     histstd  - the simulation is considered to be converged if the vlaue in each bin is
                                much larger than the standard deviation of the growth rate of the value in that bin
        scheme - Which WL scheme to run
                 inverse_time - When a histogram is converged divide the modification factor by 2 and then rerun the calculation
                 fixed_f - Stop when the convergence criteria is met. This scheme typically have to be run multiple times
                           and then the results should be averaged
        """
        self.logger = logging.getLogger( "WangLandauSGC" )
        self.logger.setLevel( logging.DEBUG )
        ch = logging.StreamHandler()
        ch.setLevel( logging.INFO )
        self.logger.addHandler(ch)

        self.atoms = None
        self.initialized = False
        self.site_types = site_types
        self.site_elements = site_elements
        self.possible_swaps = []
        self.chem_pot = {}


        self.Nbins = Nbins
        self.Emin = 500.0
        self.Emax = 510.0
        # Track the max energy ever and smallest. Important for the update range function
        self.largest_energy_ever = -np.inf
        self.smallest_energy_ever = np.inf
        self.f = initial_f
        self.f0 = initial_f
        self.flatness_criteria = flatness_criteria
        self.atoms_count = {}
        self.current_bin = 0
        self.fmin = 1E-6
        self.db_name = db_name
        self.db_id = db_id
        self.chem_pot_db_uid = {}
        self.read_params()
        self.initialize()
        self.has_found_at_least_one_structure_within_range = False
        all_schemes = ["fixed_f","inverse_time"]
        all_conv_checks = ["flathist", "histstd"]
        if ( not conv_check in all_conv_checks ):
            raise ValueError( "conv_check has to be one of {}".format(all_conv_checks) )
        if ( not scheme in all_schemes ):
            raise ValueError( "scheme hsa to be one of {}".format(all_schemes) )
        self.conv_check = conv_check
        self.scheme = scheme
        self.converged = False
        self.iter = 1
        self.check_convergence_every = 1000
        self.rejected_below = 0
        self.rejected_above = 0
        self.is_first_step = True
        all_ensambles = ["canonical","semi-grand-canonical"]
        if ( not ensemble in all_ensambles ):
            raise ValueError( "Ensemble has to one of {}".format(all_ensambles) )
        self.ensemble = ensemble

        # Some variables used to monitor the progress
        self.prev_number_of_converged = 0
        self.prev_number_of_known_bins = 0
        self.n_steps_without_progress = 0
        self.mod_factor_updater = mfu.ModificationFactorUpdater(self)
        self.on_converged_hist = chp.ConvergedHistogramPolicy(self)
        self.histogram = Histogram( self.Emin, self.Emax, self.Nbins )

        if ( scheme == "inverse_time" ):
            self.mod_factor_updater = mfu.InverseTimeScheme(self,fmin=fmin)
            self.on_converged_hist = chp.LowerModificationFactor(self,fmin=fmin)

        if (len(self.atoms) != len(self.site_types )):
            raise ValueError( "A site type for each site has to be specified!" )

        if ( not len(self.site_elements) == np.max(self.site_types)+1 ):
            raise ValueError( "Elements for each site type has to be specified!" )

        # Check that a chemical potential have been given to all elements
        if ( self.ensemble == "semi-grand-canonical" ):
            for site_elem in self.site_elements:
                for elem in site_elem:
                    if ( not elem in self.chem_pot.keys() ):
                        raise ValueError("A chemical potential for {} was not specified".format(elem) )

        self.logger.info( "Wang Landau class initialized" )
        current_time = time.localtime()
        self.logger.info( "Simulation starts at: %s"%(time.strftime("%Y-%m-%d %H:%M:%S",current_time)))
        self.logger.info( "WL-scheme: {}".format(self.scheme) )
        self.logger.info( "Convergence check: {}".format(conv_check) )
        self.logger.info( "Initial modification factor,f: {}".format(self.f))
        self.logger.info( "fmin: {} (only relevant if the scheme changes the modification factor f)".format(fmin))
        self.logger.info( "Checking convergence every {}".format(self.check_convergence_every))
        self.logger.info( "Number of bins: {}".format(self.Nbins) )
        self.logger.info( "Emin: {} eV".format(self.Emin) )
        self.logger.info( "Emax: {} eV".format(self.Emax) )

    def initialize( self ):
        """
        Constructs the site elements and site types if they are not given
        """
        if ( self.site_types is None or self.site_elements is None ):
            self.site_types = [0 for _ in range(len(self.atoms))]
            symbols = []
            for atom in symbols:
                if ( not atom.symbol in symbols ):
                    symbols.append( atom.symbol )
            self.site_elements = [symbols]

        # Count number of elements
        self.atoms_count = {key:0 for key in self.chem_pot.keys()}
        for atom in self.atoms:
            if ( atom.symbol in self.atoms_count.keys() ):
                self.atoms_count[atom.symbol] += 1
            else:
                self.atoms_count[atom.symbol] = 1

        if ( len(self.possible_swaps) == 0 ):
            self.possible_swaps = []
            for i in range(len(self.site_elements)):
                self.possible_swaps.append({})
                for element in self.site_elements[i]:
                    self.possible_swaps[-1][element] = []
                    for e in self.site_elements[i]:
                        if ( e == element ):
                            continue
                        self.possible_swaps[-1][element].append(e)

    def read_params( self ):
        """
        Reads the entries from a database
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT fmin,current_f,initial_f,queued,flatness,initialized,atomID,n_iter from simulations where uid=?", (self.db_id,) )
        entries = cur.fetchone()
        conn.close()

        self.fmin = float( entries[0] )
        self.f = float( entries[1] )
        self.f0 = float( entries[2] )

        queued = entries[3]
        self.flatness_criteria = entries[4]
        self.initialized = entries[5]
        atomID = int(entries[6])
        self.iter = int(entries[7])

        db = connect( self.db_name )
        self.atoms = db.get_atoms( id=atomID, attach_calculator=True )

        try:
            row = db.get( id=atomID )
            elms = row.data.elements
            chem_pot = row.data.chemical_potentials
            self.chem_pot = wltools.key_value_lists_to_dict(elms,chem_pot)
        except Exception as exc:
            self.logger.warning( str(exc) )

    def get_bin( self, energy ):
        """
        Returns the binning corresponding to the energy
        """
        return int( (energy-self.Emin)*self.Nbins/(self.Emax-self.Emin) )

    def get_energy( self, indx ):
        """
        Returns the energy corresponding to a given bin
        """
        return self.Emin + (self.Emax-self.Emin )*indx/self.Nbins

    def get_trial_move_sgc():
        """
        Returns a trial move consitent with the Semi Grand Canonical ensemble
        """
        indx = np.random.randint(low=0,high=len(self.atoms))
        symb = self.atoms[indx].symbol

        site_type = self.site_types[indx]
        possible_elements = self.possible_swaps[site_type][symb]

        # This is slow and should be optimized
        new_symbol = possible_elements[np.random.randint(low=0,high=len(possible_elements))]
        system_changes = [(indx,symb,new_symbol)]
        return system_changes

    def get_trial_move_canonical():
        """
        Returns a trial move consistent with the canonical ensemble
        """
        indx = np.random.randint(low=0,high=len(self.atoms))
        symb = self.atoms[indx].symbol
        site_type = self.site_types[indx]

        trial_symb = symb
        trial_site_type = -1
        indx2 = -1
        while ( trial_symb == symb or trial_site_type != site_type ):
            indx2 = np.random.randint(low=0,high=len(self.atoms))
            trial_symb = self.atoms[indx].symbol
            trial_site_type = self.site_type[indx2]

        system_changes = [(indx,symb,trial_symb),(indx2,trial_symb,symb)]
        return system_changes

    def _step( self, ignore_out_of_range=True ):
        """
        Perform one MC step
        """
        if ( ensemble == "semi-grand-canonical" ):
            system_changes = self.get_trial_move_sgc()
        elif ( ensemble == "canonical" ):
            system_changes = self.get_trial_move_canonical()

        self.atoms._calc.calculate( self.atoms, ["energy"], system_changes )
        energy = self.atoms._calc.results["energy"]

        if ( self.ensemble == "semi-grand-canonical" ):
            symb = system_changes[0][1]
            new_symbol = system_changes[0][2]
            chem_pot_change = self.chem_pot[symb]*(self.atoms_count[symb]-1) + self.chem_pot[new_symbol]*(self.atoms_count[new_symbol]+1)
            energy -= chem_pot_change

        #selected_bin = self.get_bin(energy)
        # Important to Track these because when the histogram is redistributed
        # The rounding to integer values may result in an energy that has
        # been visited is set to zero.
        # So when the range is reduced it is important to not reduce it
        # below the max and min energies that have been visited
        if ( energy < self.smallest_energy_ever or self.smallest_energy_ever==np.inf ):
            self.smallest_energy_ever = energy
        if ( energy > self.largest_energy_ever or self.largest_energy_ever==-np.inf ):
            self.largest_energy_ever = energy

        if ( energy < self.Emin ):
            if ( ignore_out_of_range ):
                self.rejected_below += 1
                self.atoms._calc.undo_changes()
                return
            self.redistribute_hist(energy,self.Emax)
        elif ( energy >= self.Emax ):
            if ( ignore_out_of_range ):
                self.rejected_above += 1
                self.atoms._calc.undo_changes()
                return
            self.redistribute_hist(self.Emin,energy)

        # Update the modification factor
        self.mod_factor_updater.update()

        selected_bin = self.get_bin(energy)
        rand_num = np.random.rand()
        diff = self.histogram.logdos[self.current_bin]-self.histogram.logdos[selected_bin]
        if ( diff > 0.0 or not self.has_found_at_least_one_structure_within_range ):
            accept_ratio = 1.0
            self.has_found_at_least_one_structure_within_range = True
        else:
            accept_ratio = np.exp( self.histogram.logdos[self.current_bin]-self.histogram.logdos[selected_bin] )

        if ( rand_num < accept_ratio  ):
            self.current_bin = selected_bin
            self.atoms_count[symb] -= 1
            self.atoms_count[new_symbol] += 1
            self.atoms._calc.clear_history()
        else:
            self.atoms._calc.undo_changes()

        self.histogram.update( self.current_bin, self.f )
        self.iter += 1

    def save( self, fname ):
        """
        Saves the object as a pickle file
        """
        with open(fname,'wb') as ofile:
            pkl.dump(self,ofile)

        base = fname.split(".")[0]
        jsonfname = base+".json"
        self.save_results(jsonfname)
        self.save_db()

    def save_results( self, fname ):
        """
        Saves the DOS to a JSON file
        """
        data = {}
        data["dos"] = self.dos.astype(float).tolist()
        data["sgc_energy"] = self.E.astype(float).tolist()
        data["f"] = float( self.f )
        data["initial_f"] = float( self.f0 )
        data["flatness_criteria"] = self.flatness_criteria
        data["histogram"] = self.histogram.astype(int).tolist()
        data["fmin"] = float( self.fmin )
        data["converged"] = int( (self.f < self.fmin ) )
        data["chem_pot"] = self.chem_pot

        with open( fname, 'w' ) as outfile:
            str_ = json.dumps(data, indent=4, sort_keys=True, separators=(",",":"), ensure_ascii=False )
            outfile.write(str_)

    def save_db( self ):
        """
        Updates the database with the entries
        """
        self.histogram.save( self.db_name, self.db_id )
        conn = sq.connect( self.db_name )
        cur = conn.cursor()

        cur.execute( "update simulations set fmin=?, current_f=?, initial_f=?, converged=? where uid=?", (self.fmin,self.f,self.f0,self.converged,self.db_id) )
        cur.execute( "update simulations set initialized=?, struct_file=? where uid=?", (1,self.struct_file,self.db_id) )
        cur.execute( "update simulations set gs_energy=? where uid=?", (self.smallest_energy_ever,self.db_id))
        cur.execute( "update simulations set n_iter=? where uid=?", (self.iter,self.db_id) )
        cur.execute( "update simulations set ensemble=? where uid=?", (self.ensemble,self.db_id))
        conn.commit()
        conn.close()
        self.logger.info( "Results saved to database {} with ID {}".format(self.db_name,self.db_id) )


    def redistribute_hist( self, Emin, Emax ):
        """
        Redistributes the histograms
        """
        old_energy = self.histogram.get_energy( self.current_bin )
        self.histogram.redistribute_hist( Emin, Emax )
        self.current_bin = self.histogram.get_bin(old_energy)

    def update_range( self ):
        """
        Updates the range
        """
        old_energy = self.histogram.get_energy( self.current_bin )
        self.histogram.update_range()
        self.current_bin = self.histogram.get_bin(old_energy)

    def set_queued_flag_db( self ):
        """
        Sets the queued flag to true in the database
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "UPDATE simulations set queued=1 WHERE uid=?", (self.db_id,) )
        conn.commit()
        conn.close()

    def has_converged( self ):
        """
        Check if the simulation has converged
        """
        if ( self.conv_check == "flathist" ):
            return self.histogram.is_flat()
        elif ( self.conv_check == "histstd" ):
            return self.histogram.std_check()
        else:
            raise ValueError("Unknown convergence check!")

    def explore_energy_space( self, nsteps=200 ):
        """
        Run Wang-Landau with a high modification factor to rapidly explore the space
        and find the energy boundaries
        """
        old_f = self.f
        self.f = np.exp(4.0)
        for i in range(nsteps):
            self._step( ignore_out_of_range=False )
            if ( (i+1)%int(nsteps/5) == 0 ):
                self.update_range()
        self.update_range()
        self.f = old_f
        self.logger.info("Selected range: Emin: {}, Emax: {}".format(self.Emin,self.Emax))


    def run( self, maxsteps=10000000 ):
        if ( self.initialized == 0 ):
            raise ValueError( "The current DB entry has not been initialized!" )
        f_small_enough = False
        self.set_queued_flag_db()

        for i in range(1,maxsteps):
            self._step( ignore_out_of_range=True )

            if ( i%self.check_convergence_every == 0 ):
                if ( self.has_converged() ):
                    status = self.on_converged_hist()
                    if ( status == SimulationState.CONVERGED ):
                        self.converged = True
                        self.logger.info( "DOS has converged" )
                        break
        self.logger.info( "Simulation ended with a modification factor of {}".format(self.f) )

    def plot_dos( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, self.dos, ls="steps" )
        ax.set_yscale("log")
        ax.set_xlabel( "SGC energy (eV)" )
        ax.set_ylabel( "Density of states" )
        return fig

    def plot_histogram( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, self.histogram, ls="steps" )
        ax.set_xlabel( "SGC energy (eV)" )
        ax.set_ylabel( "Number of times visited")
        return fig

    def plot_growth_fluctuation( self ):
        """
        Creates a plot of the growth fluctuation
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.E, self.get_growth_fluctuation(), ls="steps" )
        ax.set_xlabel( "SGC energy (eV)" )
        ax.set_ylabel( "Growth fluctuation")
        return fig
