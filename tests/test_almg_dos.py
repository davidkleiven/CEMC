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
from mcmc import sa_canonical

wl_db_name = "data/almg_canonical_dos_10x10x10.db"

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

mg_concentation = 0.1
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
    Emin,Emax = find_max_min_energy(atoms)
    db = connect( wl_db_name )
    uid = db.write( atoms )
    db.update( uid, Emin=Emin, Emax=Emax, mg_conc=mg_conc )

def init_WL_run():
    manager = wldbm.WangLandauDBManager( wl_db_name )
    db = connect( wl_db_name )
    for row in db.select():
        Emin = row.get("Emin")
        Emax = row.get("Emax")
        if ( Emin is None or Emax is None ):
            print( "Energy range for atoms object is not known!" )
            continue
        manager.insert( row.id, Emin=Emin, Emax=Emax )

def find_max_min_energy( atoms ):
    temperatures = np.linspace(5.0,1000.0,20)[::-1]
    sa = sa_canonical.SimmulatedAnnealingCanonical( atoms, temperatures, mode="minimize" )
    sa.run( steps_per_temp=50000 )
    Emin = sa.extremal_energy
    sa = sa_canonical.SimmulatedAnnealingCanonical( atoms, temperatures, mode="maximize" )
    sa.run( steps_per_temp=50000 )
    Emax = sa.extremal_energy
    print ( "Maximal energies {},{}".format(Emin,Emax) )
    return Emin,Emax

def run( runID, explore=False ):
    sum_eci = 0.0
    for key,value in ecis.iteritems():
        sum_eci += np.abs(value)

    atoms = get_atoms( mg_concentation )
    view(atoms)
    wl = wang_landau_scg.WangLandauSGC( atoms, wl_db_name, runID, conv_check="flathist", scheme="square_root_reduction", ensemble="canonical", fmin=1E-8, Nbins=1000 )

    if ( explore ):
        wl.explore_energy_space( nsteps=20000 )
        Emin = wl.histogram.Emin
        Emax = wl.histogram.Emax
        delta = Emax-Emin
        center = 0.5*(Emax+Emin)
        Emin = center - 0.5*delta
        Emax = center + 0.5*delta
        wl.histogram.Emin = Emin
        wl.histogram.Emax = Emax
        print ("Emin %.2f, Emax %.2f"%(Emin,Emax))
        print ("Exploration finished!")
    #wl.histogram.Emin = -1.12907672419
    #wl.histogram.Emax = -0.845126849602
    wl.run_fast_sampler( maxsteps=int(1E9), mode="adaptive_windows", minimum_window_width=100 )
    #wl.run( maxsteps=int(1E8) )
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
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1,1,1)

    T = np.linspace(10.0,900.0,300)
    print (len(analyzers))
    for num in range(0,len(analyzers)):
        #analyzers[num].update_dos_with_polynomial_tails( factor_low=1.5, order=2, fraction=0.2 )
        analyzers[num].normalize_dos_by_infinite_temp_limit()
        internal_energy = np.array( [analyzers[num].internal_energy(temp) for temp in T] )
        heat_capacity = np.array( [analyzers[num].heat_capacity(temp) for temp in T] )
        free_energy = np.array( [analyzers[num].free_energy(temp) for temp in T] )
        free_energy *= units.kJ*1000.0/units.mol
        internal_energy *= units.kJ*1000.0/units.mol
        heat_capacity *= units.kJ*1E6/units.mol
        entrop = np.array( [analyzers[num].entropy(temp) for temp in T] )
        entrop *= units.kJ*1E6/units.mol
        ax1.plot( T, internal_energy, label="{}".format(analyzers[num].get_chemical_formula() ))
        ax2.plot( T, heat_capacity, label="{}".format(analyzers[num].get_chemical_formula() ))
        ax3.plot( T, free_energy, label="{}".format(analyzers[num].get_chemical_formula() ))
        ax4.plot( T, entrop, label="{}".format(analyzers[num].get_chemical_formula() ))
        analyzers[num].plot_dos()

    ax1.set_xlabel( "Temperature (K)" )
    ax1.set_ylabel( "Internal energy (J/mol)" )
    ax2.set_xlabel( "Temperature (K)" )
    ax2.set_ylabel( "Heat capacity (mJ/K mol)")
    ax3.set_xlabel( "Temperature (K)" )
    ax3.set_ylabel( "Helmholtz Free Energy (J/mol)")
    ax4.set_xlabel( "Temperature (K)" )
    ax4.set_ylabel( "Entropy per atom (mJ/K)")
    ax1.legend( loc="best", frameon=False )
    ax2.legend( loc="best", frameon=False )
    ax3.legend( loc="best", frameon=False )
    ax4.legend( loc="best", frameon=False )
    num = 0
    new_temps = [100,300,500,800]
    analyzers[num].plot_degree_of_contribution(new_temps)
    plt.show()


def main( mode ):
    if ( mode == "add" ):
        add_new( mg_concentation )
    elif ( mode == "initWL" ):
        init_WL_run()
    elif ( mode == "run" ):
        run(0,explore=False)
    elif ( mode == "analyze" ):
        analyze()
    elif( mode == "limits" ):
        find_max_min_energy()

if __name__ == "__main__":
    mode = sys.argv[1]
    main(mode)
