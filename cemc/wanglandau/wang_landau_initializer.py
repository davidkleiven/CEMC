from ase.db import connect
from cemc.wanglandau import get_ce_calc
from ase.ce import BulkCrystal, BulkSpacegroup
from cemc.mcmc import SimmulatedAnnealingCanonical
from ase.calculators.singlepoint import SinglePointCalculator
import pickle as pck

class WangLandauInit(object):
    def __init__( self, wl_db_name ):
        self.wl_db = wl_db_name

    def insert_atoms( bc_kwargs, size=[1,1,1], composition=None, cetype="BulkCrystal", \
                      T=None, n_steps_per_temp=10000, eci=None ):
        """
        Insert a new atoms object into the database
        """
        if ( small_bc is None ):
            raise TypeError( "No bulk crystal object given!" )

        if ( composition is None ):
            raise TypeError( "No composition given" )
        allowed_ce_types = ["BulkCrystal","BulkSpacegroup"]
        if ( not cetype in allowed_ce_types ):
            raise ValueError( "cetype has to be one of {}".format(allowed_ce_types) )
        self.cetype = ctype

        if ( eci is None ):
            raise ValueError( "No ECIs given! Cannot determine required energy range!")

        if ( cetype == "BulkCrystal" ):
            small_bc = BulkCrystal(**bc_kwargs )
        elif( ctype == "BulkSpacegroup" ):
            small_bc = BulkSpacegroup(**bc_kwargs)


        calc = get_ce_calc( small_bc, bc_kwargs, eci=eci, size=size )
        calc.set_composition( composition )
        bc = calc.bc
        bc.atoms.set_calculator(calc)

        Emin, Emax = self._find_energy_range( bc.atoms, T, nsteps_per_temp )
        cf = calc.get_cf()
        data = {"cf":cf}
        # Store this entry into the database
        db = connect( self.wl_db )
        scalc = SinglePointCalculator( bc.atoms, energy=Emin )
        bc.atoms.set_calculator(scalc)

        outfname = "BC_wanglandau_{}.pkl".format( bc.atoms.get_chemical_formula() )
        with open( outfname, 'wb' ) as outfile:
            pck.dump( bc, outfile )

        kvp = {"Emin":Emin,"Emax":Emax,"bcfile":outfname,"cetype":self.cetype}
        data["bc_kwargs"] = bc_kwargs
        data["supercell_size"] = size
        db.write( calc.BC.atoms, key_value_paris=kvp, data=data )

    def _find_energy_range( self, atoms, T, nsteps_per_temp ):
        """
        Finds the maximum and minimum energy by Simulated Annealing
        """
        print ( "Finding minimum energy")
        sa = SimmulatedAnnealingCanonical( atoms, T, mode="minimize" )
        sa.run( steps_per_temp=nsteps_per_temp )
        Emin = sa.extremal_energy
        sa = SimmulatedAnnealingCanonical( atoms, T, mode="maximize" )
        print ("Finding maximum energy.")
        sa.run( steps_per_temp=nsteps_per_temp )
        Emax = sa.extremal_energy
        print ("Minimum energy: {}, Maximum energy: {}".format(Emin,Emax))
        return Emin,Emax

    def prepare_wang_landau_run( self, select_cond, wl_kwargs ):
        """
        Prepares a Wang Landau run
        """
        manager = WangLandauDBManager( self.wl_db_name )
        atomID = db.get( select_cond ).id
        manager.insert( atomID, **wl_kwargs )

    def get_atoms( self, atomID, eci ):
        """
        Returns an instance of the atoms object requested
        """
        db = connect( self.db_name )
        row = db.get(id=atomID)
        bcfname = row.key_value_paris["bc_fname"]
        init_cf = row.data["cf"]
        try:
            with open(bcfname,'rb') as infile:
                bc = pck.load(infile)
            calc = CE( bc, eci, initial_cf=ini_cf )
            bc.atoms.set_calculator( calc )
            return bc.atoms
        except IOError as exc:
            print (str(exc))
            print ("Will try to recover the BulkCrystal object" )
            bc_kwargs = row.data["bc_kwargs"]
            cetype = row.key_value_paris["cetype"]
            if ( cetype == "BulkCrystal" ):
                small_bc = BulkCrystal( **bc_kwargs )
            else:
                small_bc = BulkSpacegroup( **bc_kwargs )
            size = row.data["supercell_size"]
            calc = get_ce_calc( small_bc, bc_kwargs, eci=eci, size=size )

            # Determine the composition
            count = row.count_atoms()
            for key in count.keys():
                count /= float( row.natoms )
            calc.set_composition( count )
            bc = calc.BC
            bc.atoms.set_calculator( calc )
            return bc.atoms
        finally:
            raise RuntimeError( "Did not manage to return the atoms object with the proper calculator attached..." )
