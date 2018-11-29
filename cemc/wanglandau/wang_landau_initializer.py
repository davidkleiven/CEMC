from ase.db import connect
from cemc import get_atoms_with_ce_calc, CE
from ase.clease import CEBulk, CECrystal
from cemc.mcmc import SimulatedAnnealingCanonical
from ase.calculators.singlepoint import SinglePointCalculator
import pickle as pck
from cemc.wanglandau.wang_landau_db_manager import WangLandauDBManager
import copy

class AtomExistsError(Exception):
    def __init__(self, msg):
        super(AtomExistsError,self).__init__(msg)

class WangLandauInit(object):
    def __init__( self, wl_db_name ):
        self.wl_db_name = wl_db_name

    def insert_atoms( self, bc_kwargs, size=[1,1,1], composition=None, cetype="CEBulk", \
                      T=None, n_steps_per_temp=10000, eci=None ):
        """
        Insert a new atoms object into the database
        """
        if ( composition is None ):
            raise TypeError( "No composition given" )
        allowed_ce_types = ["CEBulk","CECrystal"]
        if ( not cetype in allowed_ce_types ):
            raise ValueError( "cetype has to be one of {}".format(allowed_ce_types) )
        self.cetype = cetype

        if ( eci is None ):
            raise ValueError( "No ECIs given! Cannot determine required energy range!")

        if ( cetype == "CEBulk" ):
            small_bc = CEBulk(**bc_kwargs )
            small_bc.reconfigure_settings()
        elif( cetype == "CECrystal" ):
            small_bc = CECrystal(**bc_kwargs)
            small_bc.reconfigure_settings()

        self._check_eci(small_bc,eci)

        atoms = get_atoms_with_ce_calc( small_bc, bc_kwargs, eci=eci, size=size )
        calc = atoms.get_calculator()
        calc.set_composition(composition)

        formula = atoms.get_chemical_formula()
        if ( self.template_atoms_exists(formula) ):
            raise AtomExistsError( "An atom object with the specified composition already exists in the database" )

        Emin, Emax = self._find_energy_range(atoms, T, n_steps_per_temp)
        cf = calc.get_cf()
        data = {"cf":cf}
        # Store this entry into the database
        db = connect( self.wl_db_name )
        scalc = SinglePointCalculator( atoms, energy=Emin )
        atoms.set_calculator(scalc)

        outfname = "BC_wanglandau_{}.pkl".format(atoms.get_chemical_formula())
        with open( outfname, 'wb' ) as outfile:
            pck.dump( (calc.BC, atoms), outfile )

        kvp = {"Emin":Emin,"Emax":Emax,"bcfile":outfname,"cetype":self.cetype}
        data["bc_kwargs"] = small_bc.kwargs
        data["supercell_size"] = size
        db.write( atoms, key_value_pairs=kvp, data=data )

    def _check_eci(self,bc,eci):
        """
        Verify that all ECIs corresponds to a cluster
        """
        cnames = copy.deepcopy(bc.cluster_family_names)

        for key in eci.keys():
            if (key.startswith("c0") or key.startswith("c1")):
                continue
            name = key.rpartition("_")[0]
            if name not in cnames:
                raise ValueError("There are ECIs that does not fit a cluster. ECIs: {}, Cluster names: {}".format(eci,flattened))

    def _find_energy_range( self, atoms, T, nsteps_per_temp ):
        """
        Finds the maximum and minimum energy by Simulated Annealing
        """
        print ( "Finding minimum energy")
        sa = SimulatedAnnealingCanonical(atoms, T, mode="minimize")
        sa.run( steps_per_temp=nsteps_per_temp )
        Emin = sa.extremal_energy
        sa = SimulatedAnnealingCanonical(atoms, T, mode="maximize")
        print ("Finding maximum energy.")
        sa.run( steps_per_temp=nsteps_per_temp )
        Emax = sa.extremal_energy
        print ("Minimum energy: {}, Maximum energy: {}".format(Emin,Emax))
        return Emin,Emax

    def prepare_wang_landau_run( self, select_cond, wl_kwargs={} ):
        """
        Prepares a Wang Landau run
        """
        manager = WangLandauDBManager( self.wl_db_name )
        print (manager)
        db = connect( self.wl_db_name )
        atomID = db.get( select_cond ).id
        row = db.get(select_cond)
        Emin = row.Emin
        Emax = row.Emax
        wl_kwargs["Emin"] = Emin
        wl_kwargs["Emax"] = Emax
        if ( "only_new" not in wl_kwargs.keys() ):
            wl_kwargs["only_new"] = False

        if ( self.atoms_exists_in_db(atomID ) ):
            manager.add_run_to_group( atomID )
        else:
            manager.insert( atomID, **wl_kwargs )

    def atoms_exists_in_db( self, atomID ):
        """
        Check if the atoms object exists
        """
        db = connect( self.wl_db_name )
        ref_formula = db.get( id=atomID ).formula
        for row in db.select():
            if ( row.id == atomID ):
                continue
            if ( row.formula == ref_formula ):
                return True
        return False

    def template_atoms_exists( self, formula ):
        """
        Check if the template object already exists
        """
        db = connect( self.wl_db_name )
        for row in db.select():
            if ( row.formula == formula ):
                return True
        return False

    def get_atoms( self, atomID, eci ):
        """
        Returns an instance of the atoms object requested
        """
        db = connect( self.wl_db_name )
        row = db.get(id=atomID)
        bcfname = row.key_value_pairs["bcfile"]
        init_cf = row.data["cf"]
        try:
            with open(bcfname,'rb') as infile:
                bc, atoms = pck.load(infile)
            calc = CE(atoms, bc, eci, initial_cf=init_cf)
            return atoms
        except IOError as exc:
            print (str(exc))
            print ("Will try to recover the CEBulk object" )
            bc_kwargs = row.data["bc_kwargs"]
            cetype = row.key_value_pairs["cetype"]
            if ( cetype == "CEBulk" ):
                small_bc = CEBulk( **bc_kwargs )
                small_bc.reconfigure_settings()
            else:
                small_bc = CECrystal( **bc_kwargs )
                small_bc.reconfigure_settings()
            size = row.data["supercell_size"]
            atoms = get_atoms_with_ce_calc( small_bc, bc_kwargs, eci=eci, size=size )
            calc = atoms.get_calculator()

            # Determine the composition
            count = row.count_atoms()
            for key in count.keys():
                count /= float( row.natoms )
            calc.set_composition( count )
            return atoms
        except:
            raise RuntimeError( "Did not manage to return the atoms object with the proper calculator attached..." )
