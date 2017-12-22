import sys
sys.path.insert( 1,"/home/davidkl/Documents/aseJin" )
from wanglandau import wang_landau_scg
from wanglandau import wang_landau_db_manger as wldbm
from ase.build import bulk
from ase.db import connect
from wanglandau.ce_calculator import CE
from ase.ce.settings import BulkCrystal
from ase.visualize import view
import numpy as np
from matplotlib import pyplot as plt
from ase import units
wl_db_name = "data/almg_canonical_dos.db"

ecis = {'c3_1225_4_1': -0.00028826723864655595,
        'c2_1000_1_1': -0.012304759727020153,
        'c4_1225_7_1': 0.00018000893943061064,
        'c2_707_1_1': 0.01078731693580544,
        'c4_1225_3_1': 0.00085623111812932343,
        'c2_1225_1_1': -0.010814400169849577,
        'c4_1000_1_1': 0.0016577886586285448,
        'c4_1225_2_1': 0.01124654696678576,
        'c3_1225_2_1': -0.017523737495758165,
        'c4_1225_6_1': 0.0038879587131474451,
        'c4_1225_5_1': 0.00060830459771275532,
        'c3_1225_3_1': -0.011318935831421125
        }

def get_atoms( mg_conc ):
    db_name = "/home/davidkl/Documents/WangLandau/data/ce_hydrostatic_7x7.db"
    conc_args = {
        "conc_ratio_min_1":[[60,4]],
        "conc_ratio_max_1":[[64,0]],
    }
    ceBulk = BulkCrystal( "fcc", 4.05, [10,10,10], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, max_cluster_dia=1.414*4.05,reconf_db=True )
    init_cf = {key:1.0 for key in ecis.keys()}
    calc = CE( ceBulk, ecis, initial_cf=init_cf )
    ceBulk.atoms.set_calculator( calc )
    atoms = ceBulk.atoms

    n_mg = int(mg_conc*len(atoms))
    for i in range(n_mg):
        atoms._calc.update_cf( (i,"Al","Mg") )
    return atoms

def add_new( mg_conc ):
    atoms = get_atoms(mg_conc)
    db = connect( wl_db_name )
    db.write( atoms )

def init_WL_run( atomID ):
    manager = wldbm.WangLandauDBManager( wl_db_name )
    manager.insert( atomID, Nbins=1000 )

def run( runID, explore=False ):
    atoms = get_atoms(0.1)
    view(atoms)
    wl = wang_landau_scg.WangLandauSGC( atoms, wl_db_name, runID, conv_check="flathist", scheme="square_root_reduction", ensemble="canonical", fmin=1E-6, Nbins=1000 )

    if ( explore ):
        wl.explore_energy_space( nsteps=20000 )
        Emin = wl.histogram.Emin
        Emax = wl.histogram.Emax
        delta = Emax-Emin
        center = 0.5*(Emax+Emin)
        Emin = center - 0.75*delta
        Emax = center + 0.75*delta
        wl.histogram.Emin = Emin
        wl.histogram.Emax = Emax

    wl.run( maxsteps=int(1E8) )
    wl.save_db()

def analyze():
    manager =  wldbm.WangLandauDBManager(wl_db_name)
    analyzers = manager.get_analyzer_all_groups()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)

    T = np.linspace(10.0,900.0,300)
    num = 1
    for num in range(0,3):
        analyzers[num].normalize_dos_by_infinite_temp_limit()
        internal_energy = np.array( [analyzers[num].internal_energy(temp) for temp in T] )
        heat_capacity = np.array( [analyzers[num].heat_capacity(temp) for temp in T] )
        free_energy = np.array( [analyzers[num].free_energy(temp) for temp in T] )
        free_energy *= units.kJ/units.mol
        internal_energy *= units.kJ/units.mol
        heat_capacity *= units.kJ/units.mol
        ax1.plot( T, internal_energy )
        ax2.plot( T, heat_capacity )
        ax3.plot( T, free_energy )
        analyzers[num].plot_dos()

    ax1.set_xlabel( "Temperature (K)" )
    ax1.set_ylabel( "Internal energy (kJ/mol)" )
    ax2.set_xlabel( "Temperature (K)" )
    ax2.set_ylabel( "Heat capacity (kJ/K mol)")
    ax3.set_xlabel( "Temperature (K)" )
    ax3.set_ylabel( "Helmholtz Free Energy (kJ/mol)")

    new_temps = [100,300,500,800]
    analyzers[num].plot_degree_of_contribution(new_temps)
    plt.show()


def main( mode ):
    if ( mode == "add" ):
        add_new(0.1)
    elif ( mode == "initWL" ):
        init_WL_run(5)
    elif ( mode == "run" ):
        run(4,explore=True)
    elif ( mode == "analyze" ):
        analyze()

if __name__ == "__main__":
    mode = sys.argv[1]
    main(mode)
