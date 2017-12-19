import sys
sys.path.append( "/home/davidkl/Documents/WangLandau/wanglandau" )
from simple_ce_calc import CEcalc
from ce_calculator import CE
from ase.build import bulk
from ase.ce.settings import BulkCrystal
from sa_sgc import SimmualtedAnnealingSGC
from ase.visualize import view
from matplotlib import pyplot as plt
from mcmc import montecarlo as mc
from mcmc import mc_observers as mc_obs
import numpy as np

# Hard coded ECIs obtained from the ce_hydrostatic.db runs
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

def mcmc( ceBulk, c_mg ):

    n_mg = int( c_mg*len(ceBulk.atoms) )
    for i in range(n_mg):
        ceBulk.atoms._calc.update_cf( (i,"Al","Mg") )
    ceBulk.atoms._calc.clear_history()
    mc_obj = mc.Montecarlo( ceBulk.atoms, 2000.0 )

    # Run Monte Carlo
    obs_pre = mc_obs.CorrelationFunctionTracker( ceBulk.atoms._calc )
    mc_obj.attach( obs_pre )
    tot_en_1 = mc_obj.runMC( steps=1000 )
    obs_pre.plot_history( max_size=3 )
    plt.show()
    view( ceBulk.atoms )

    observer = mc_obs.CorrelationFunctionTracker( ceBulk.atoms._calc )
    mc_obj = mc.Montecarlo( ceBulk.atoms, 10.0 )
    #mc_obj.T = 10.0
    mc_obj.attach( observer, interval=1 )
    tot_en = mc_obj.runMC( steps=100000 )
    view( ceBulk.atoms )
    observer.plot_history(max_size=3)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( tot_en )
    ax.plot( np.cumsum(tot_en)/(np.arange(len(tot_en))+1.0) )
    ax.plot( tot_en_1 )
    plt.show()

def main( run ):
    atoms = bulk("Al")
    atoms = atoms*(4,4,4)
    atoms[0].symbol = "Mg"

    db_name = "/home/davidkl/Documents/WangLandau/data/ce_hydrostatic.db"
    conc_args = {
        "conc_ratio_min_1":[[60,4]],
        "conc_ratio_max_1":[[64,0]],
    }
    ceBulk = BulkCrystal( "fcc", 4.05, [7,7,7], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False )
    init_cf = {key:1.0 for key in ecis.keys()}

    calc = CE( ceBulk, ecis, initial_cf=init_cf )
    ceBulk.atoms.set_calculator( calc )

    if ( run == "MC" ):
        mcmc( ceBulk, 0.1 )
    else:
        chem_pot = {
        "Al":0.0,
        "Mg":0.0
        }
        ecis["c1_1"] = 0.025# Change the single particle interaction term to mimic a chemical potential
        gs_finder = SimmualtedAnnealingSGC( ceBulk.atoms, chem_pot, "test_db.db" )
        gs_finder.run( n_steps=1000, Tmin=400, ntemps=10 )
        gs_finder.show_visit_stat()
        gs_finder.show_compositions()
    view( ceBulk.atoms )
    #plt.show()

if __name__ == "__main__":
    run = sys.argv[1] # Run is WL or MC
    main( run )
