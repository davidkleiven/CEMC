import sys
from ase.calculators.emt import EMT
from ase.build import bulk
from wang_landau_scg import WangLandauSGC
from matplotlib import pyplot as plt
import pickle as pkl
from sgc_to_cannonical import SGCToCanonicalConverter
from wang_landau_db_manger import WangLandauDBManger
import numpy as np
from sa_sgc import SimmualtedAnnealingSGC

db_name = "/home/davidkl/Documents/GPAWTutorial/CE_extensions/WangLandau/wl_au_cu_1000bins.db"
def plot_probablility_distribution(wl):
    T = [100,200,300,400,500,800,1000,1200]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for temp in T:
        ax.plot( wl.E, wl.dos*wl._boltzmann_factor(temp), ls="steps", label="T={} K".format(temp) )
    ax.legend()
    ax.set_yscale("log")
    return fig

def initialize_db():
    manager = WangLandauDBManger( db_name )
    manager.prepare_from_ground_states( 900.0, initial_f=1.6, Nbins=1000, n_kbT=40 )

def find_GS():
    atoms = bulk("Au")
    atoms = atoms*(4,4,4)
    calc = EMT()
    atoms.set_calculator(calc)
    chem_pot = {
    "Cu":0.9,
    "Au":0.0
    }

    gs_finder = SimmualtedAnnealingSGC( atoms, chem_pot, db_name )
    gs_finder.run( n_steps=1000 )

def update_groups():
    manager = WangLandauDBManger( db_name )
    for i in range(1,12):
        manager.add_run_to_group(i,n_entries=10)


def analyze():
    db_manager = WangLandauDBManger( db_name )
    analyzers = db_manager.get_analyzer_all_groups()
    T = [10,100,200,300,400,500,800,1000,5000]
    analyzers[0].plot_dos()
    analyzers[0].plot_degree_of_contribution(T)
    analyzer = SGCToCanonicalConverter(analyzers,64)
    chem_pot, comp, sgc_pots, chem_pot_raw, sgc_pots_raw = analyzer.get_compositions(T[5])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( chem_pot["Cu"], sgc_pots["Cu"], label="Free energy" )
    ax.plot( chem_pot_raw["Cu"], sgc_pots_raw["Cu"], 'o', mfc="none" )
    ax.set_xlabel( "Chemical potential" )
    ax.set_ylabel( "SGC potential" )
    ax2 = ax.twinx()
    ax2.plot( chem_pot["Cu"], comp["Cu"], color="#fdbf6f" )
    ax.plot( [],[], color="#fdbf6f", label="Conc")
    ax2.set_ylabel( "Concentration Cu" )
    ax.legend( loc="best", frameon=False )

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    temps = np.linspace( 200.0, 800.0 ,5000 )
    for an in analyzers:
        mu = an.chem_pot["Cu"]
        Cv = np.array( [an.heat_capacity(t) for t in temps] )
        E = np.array( [an.internal_energy(t) for t in temps] )
        ax2.plot( temps, Cv, label="\$\mu=%.2f\$"%(mu) )
    ax2.legend( loc="best", frameon=False )
    ax2.set_xlabel( "Temperature (K)" )
    ax2.set_ylabel( "Heat capacity (J/K)" )

    x, helmholtz = analyzer.free_energy(T[5])
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.plot( x,helmholtz )
    ax3.set_xlabel( "Concentration Cu" )
    ax3.set_ylabel( "Helmholtz Free energy" )
    temps = np.linspace(200,500,100)
    analyzer.binary_diagram( temps, "Cu", "Au" )
    plt.show()

def main( runID ):
    site_types = [0 for _ in range(64)]
    site_elements = [["Cu","Au"]]
    wl = WangLandauSGC( db_name, runID, site_types=site_types, site_elements=site_elements, Nbins=100, scheme="fixed_f", conv_check="histstd" )
    wl.run( maxsteps=100000 )
    #wl.explore_energy_space( nsteps=1000 )
    wl.save_db()
    #wl.plot_dos()
    #wl.plot_histogram()
    #wl.plot_growth_fluctuation()
    #plot_probablility_distribution(wl)
    #plt.show()

if __name__ == "__main__":
    opt = sys.argv[1]
    if ( opt == "analyze" ):
        analyze()
    elif ( opt == "run" ):
        runID = int(sys.argv[2] )
        main(runID)
    elif ( opt == "init" ):
        initialize_db()
    elif ( opt == "add" ):
        update_groups()
    elif ( opt == "gs" ):
        find_GS()
