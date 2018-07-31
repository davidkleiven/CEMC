import sys
import sqlite3 as sq
import numpy as np
from cemc.wanglandau.wl_analyzer import WangLandauSGCAnalyzer
from cemc.wanglandau import wltools
from ase.db import connect
import ase.units
from scipy import special
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
mpl.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plt
import copy

class WangLandauDBManager( object ):
    def __init__( self, db_name ):
        self.db_name = db_name
        self.check_db()

    def check_db( self ):
        """
        Checks if the database has the correct format and updates it if not
        """
        required_fields = {
        "simulations":["uid","logdos","energy","histogram","fmin","current_f","initial_f","converged",
        "queued","Nbins","flatness","growth_variance","Emin","Emax","initialized","struct_file",
        "gs_energy","atomID","n_iter","ensemble","known_structures"],
        "chemical_potentials":["uid","element","id","potential"]
        }
        required_tables = ["simulations"]
        types = {
            "id":"interger",
            "logdos":"blob",
            "energy":"blob",
            "histogram":"blob",
            "fmin":"float",
            "current_f":"float",
            "initial_f":"float",
            "converged":"integer",
            "element":"text",
            "potential":"float",
            "Nbins":"integer",
            "flatness":"float",
            "queued":"integer",
            "growth_variance":"blob",
            "Emin":"float",
            "Emax":"float",
            "initialized":"integer",
            "struct_file":"text",
            "gs_energy":"float",
            "atomID":"integer",
            "n_iter":"integer",
            "ensemble":"text",
            "known_structures":"blob"
        }

        conn = sq.connect( self.db_name )
        cur = conn.cursor()

        default_values = {
            "converged":0
        }

        for tabname in required_tables:
            sql = "create table if not exists %s (uid integer)"%(tabname)
            cur.execute(sql)
        conn.commit()

        # Check if the tables has the required fields
        for tabname in required_tables:
            for col in required_fields[tabname]:
                try:
                    sql = "alter table %s add column %s %s"%(tabname, col, types[col] )
                    if ( col in default_values ):
                        sql += " DEFAULT({})".format(default_values[col])
                    cur.execute( sql )
                except Exception as exc:
                    pass
        conn.commit()
        conn.close()

    def get_new_id( self ):
        """
        Get new ID in the simulations table
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute("SELECT uid FROM simulations" )
        ids = list(cur.fetchall())
        ids = [item[0] for item in ids]
        conn.close()
        print(ids)
        if len(ids) == 0:
            return 0
        return np.max(ids)+1

    def prepare_from_ground_states( self, Tmax=300.0, initial_f=2.71, fmin=1E-8, flatness=0.8, Nbins=50, n_kbT=20 ):
        """
        Create one Wang-Landau simulation from all ground state structures
        """
        # Extract all atomsIDs already in the database
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT atomID FROM simulations" )
        entries = cur.fetchall()
        conn.close()
        atIds = [entry[0] for entry in entries]

        db = connect( self.db_name )
        for row in db.select():
            if ( row.id in atIds[0] ):
                continue
            self.insert(
                row.id, Tmax, initial_f=initial_f, fmin=fmin,
                flatness=flatness, Nbins=Nbins, n_kbT=n_kbT)

    def exists_in_db( self, atomID ):
        """
        Check if the object is already present in the database
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT atomID FROM simulations" )
        entries = cur.fetchall()
        conn.close()
        exists = False
        for atID in entries:
            if ( atomID == atID[0] ):
                return True
        return False


    def insert( self, atomID, initial_f=2.71, fmin=1E-8, flatness=0.8, Nbins=50, Emin=0.0, Emax=1.0, only_new=True ):
        """
        Insert a new entry into the database
        """
        newID = self.get_new_id()
        if ( self.exists_in_db(atomID) and only_new ):
            print ("A WL simulation of the atomID already exists in the database" )
            return
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "insert into simulations (uid,initial_f,current_f,flatness,fmin,queued,Nbins,atomID) values (?,?,?,?,?,?,?,?)",
        (newID, initial_f,initial_f,flatness,fmin,0,Nbins,atomID) )
        cur.execute( "update simulations set initialized=? where uid=?", (0,newID) )
        cur.execute( "update simulations set n_iter=? where uid=?", (1,newID) )
        conn.commit()

        cur.execute( "update simulations set Emin=?, Emax=? where uid=?",(Emin,Emax,newID))
        E = np.linspace( Emin,Emax+1E-8,Nbins )
        cur.execute( "update simulations set energy=? where uid=?", (wltools.adapt_array(E),newID) )
        cur.execute( "update simulations set initialized=? where uid=?", (1,newID) )
        conn.commit()
        conn.close()

    def get_energy_range( self, atomID, Tmax, n_kbT ):
        """
        Computes the energy range based on the ground state of the atom
        """
        db = connect( self.db_name )
        try:
            row = db.get( id=atomID )
            elms = row.data.elements
            chem_pot = row.data.chemical_potentials
            Emin = row.energy
            chem_pot = wltools.key_value_lists_to_dict( elms, chem_pot )
            at_count = wltools.element_count( db.get_atoms(id=atomID) )


            for key,value in chem_pot.items():
                if ( not key in at_count.keys() ):
                    continue
                Emin -= value*at_count[key]

            Emax = Emin + n_kbT*ase.units.kB*Tmax
        except:
            Emin = 0.0
            Emax = 100.0
        return Emin,Emax

    def add_run_to_group( self, atomID, n_entries=1 ):
        """
        Adds a run to a group
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT uid,initial_f,current_f,flatness,fmin,queued,Nbins,atomID,Emin,Emax,initialized,energy,gs_energy FROM simulations WHERE atomID=?", (atomID,))
        entries = cur.fetchone()
        oldID = int(entries[0])
        newID = self.get_new_id()
        entries = list(entries)

        for _ in range(n_entries):
            entries[0] = newID
            entries[7] = atomID
            entries[5] = 0 # Set queued flag to 0
            entries = tuple(entries)
            cur.execute( "INSERT INTO simulations (uid,initial_f,current_f,flatness,fmin,queued,Nbins,atomID,Emin,Emax,initialized,energy,gs_energy) values (?,?,?,?,?,?,?,?,?,?,?,?,?)", entries )
            entries = list(entries)
            newID += 1
        conn.commit()
        conn.close()

    def get_converged_wl_objects( self, atoms, calc ):
        """
        Get a list of all converged Wang-Landau simulations
        """
        conn = sq.connect( self.db_name )
        cur.execute( "SELECT UID FROM simulations WHERE converged=1" )
        uids = cur.fetchall()
        conn.close()
        return self.get_wl_objects( atoms, calc, uids )

    def get_wl_objects( self, atoms, calc, uids ):
        """
        Returns a list of Wang Landau objects corresponding to ids
        """
        obj = []
        for uid in uids:
            objs.append( WangLandauSGC( atoms, calc, self.db_name, uid ) )

    def get_analyzer( self, atomID, min_number_of_converged=1 ):
        """
        Returns a Wang-Landau Analyzer object based on the average of all converged runs
        within a atomID
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT energy,logdos,uid,ensemble,known_structures FROM simulations WHERE converged=1 AND atomID=?", (atomID,) )
        entries = cur.fetchall()
        conn.close()

        if ( len(entries) == 0 ):
            return None
        all_logdos = None
        for entry in entries:
            try:
                uid = int( entry[2] )
                energy = wltools.convert_array( entry[0] )
                logdos = wltools.convert_array( entry[1] )
                known_states = wltools.convert_array( entry[4] ).astype(np.uint8)
                ensemble = entry[3]
            except:
                continue

            logdos = logdos[known_states==1]
            energy = energy[known_states==1]
            #plt.plot(logdos,ls="steps")
            #plt.show()
            logdos -= np.max(logdos) # Avoid overflow
            logdos += 10.0
            ref_e0 = logdos[0]
            if ( all_logdos is None ):
                all_logdos = copy.deepcopy( logdos )
            else:
                all_logdos += logdos

            if ( ensemble == "canonical" ):
                chem_pot = None
            elif ( ensemble == "semi-grand-canonical" ):
                db = connect( self.db_name )
                row = db.get( id=atomID )
                elms = row.data.elements
                pots = row.data.chemical_potentials
                chem_pot = wltools.key_value_lists_to_dict( elms, pots )
                gs_atoms = db.get_atoms( id=atomID )
                count = wltools.element_count(gs_atoms)
            else:
                raise ValueError( "Unknown statistical ensemble. Got {}".format(ensemble) )

        db = connect( self.db_name )
        row = db.get(id=atomID)
        dos = np.exp(all_logdos/len(entries))
        return WangLandauSGCAnalyzer( energy, dos, row.numbers, chem_pot=chem_pot )

    def get_analyzer_all_groups( self  ):
        """
        Returns a list of analyzer objects
        """
        analyzers = []
        db = connect( self.db_name )
        for row in db.select():
            new_analyzer = self.get_analyzer( row.id )
            analyzers.append( new_analyzer )
        filtered = [entry for entry in analyzers if not entry is None] # Remove non-converged entries
        return filtered

    def get_next_non_converged_uid( self, atomID ):
        """
        Returns the UID of the next non-converged entry
        """
        conn = sq.connect( self.db_name )
        cur = conn.cursor()
        cur.execute( "SELECT uid FROM simulations WHERE atomID=? AND converged=?",(atomID,0) )
        next_uid = cur.fetchone()[0]
        conn.close()
        if ( next_uid is None ):
            return -1
        return next_uid
