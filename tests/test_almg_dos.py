import sys
from wanglandau import wang_landau_scg
from wanglandau import wang_landau_db_manager
from ase.build import bulk
from ase.db import connect
from wanglandu import ce_calculator
db_name = "data/almg_canonical_dos.db"

ecis = {'c3_1225_4_1': -0.00028826723864655595,
        'c2_1000_1_1': -0.012304759727020153,
        'c4_1225_7_1': 0.00018000893943061064,
        'c2_707_1_1': 0.01078731693580544,
        'c4_1225_3_1': 0.00085623111812932343,
        'c2_1225_1_1': -0.010814400169849577,
        'c1_1': -1.0666948263880078,
        'c4_1000_1_1': 0.0016577886586285448,
        'c4_1225_2_1': 0.01124654696678576,
        'c3_1225_2_1': -0.017523737495758165,
        'c4_1225_6_1': 0.0038879587131474451,
        'c4_1225_5_1': 0.00060830459771275532,
        'c3_1225_3_1': -0.011318935831421125,
        u'c0': -2.6466290360293874}

def add_new( mg_conc ):
    calc = ce_calculator.CE
    db_name = "/home/davidkl/Documents/WangLandau/data/ce_hydrostatic_20x20.db"
    conc_args = {
        "conc_ratio_min_1":[[60,4]],
        "conc_ratio_max_1":[[64,0]],
    }
    ceBulk = BulkCrystal( "fcc", 4.05, [7,7,7], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, max_cluster_dia=1.414*4.05,reconf_db=False )
    calc = CE( ceBulk, ecis, initial_cf=init_cf )
    ceBulk.atoms.set_calculator( calc )
    atoms = ceBulk.atoms

    n_mg = int(mg_conc*len(atoms))
    for i in range(n_mg):
        atoms[i].symbol = "Mg"
    db = connect( db_name )
    db.write(atoms)

def init_WL_run( atomID ):
    manager = wang_landau_db_manager.WangLandauDBManager( db_name )
    manager.insert( atomID, Nbins=100 )

def run( runID, explore=False ):
    wl = wang_landau_sgc.WangLandauSGC( db_name, runID, conv_check="histstd", scheme="inverse_time", ensemble="canonical" )

    if ( explore ):
        wl.explore_energy_space( nsteps=20000 )

    wl.run( nsteps=1E7 )
    wl.save_db()

def main( mode ):
    if ( mode == "add" ):
        add_new(0.1)
    elif ( mode == "initWL" ):
        init_WL_run(1)
    elif ( mode == "run" ):
        run(0,explore=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    main(mode)
