import sys
from wanglandau.ce_calculator import CE
from ase.build import bulk
from ase.clease.settings import CEBulk
from wanglandau.sa_sgc import SimmualtedAnnealingSGC
from ase.visualize import view
from mcmc import montecarlo as mc
from mcmc import mc_observers as mc_obs
from ase.units import kB
import numpy as np
import json
import copy

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
        'c3_1225_3_1': -0.011318935831421125}

with open("/home/davidkl/Documents/GPAWTutorial/CE/data/almg_eci.json") as infile:
    ecis = json.load(infile)

#ecis["c2_707_1_1"] = 0.0
print (ecis)
def update_phonons( ecis, phonon_ecis, temp ):
    for key,value in phonon_ecis.items():
        if ( not key in ecis.keys() ):
            ecis[key] = 0.0

        ecis[key] += kB*temp*value
    return ecis

def mcmc( ceBulk, c_mg, phonons=False ):
    n_mg = int( c_mg*len(ceBulk.atoms) )
    for i in range(n_mg):
        ceBulk.atoms._calc.update_cf( (i,"Al","Mg") )
    ceBulk.atoms._calc.clear_history()
    formula = ceBulk.atoms.get_chemical_formula()
    out_file = "data/pair_corrfuncs_tempdependent_gs%s.json"%(formula)
    temps = [800,700,600,500,400,300,200,100]
    n_burn = 40000
    n_sampling = 100000
    cfs = []
    cf_std = []
    n_samples = []
    origin_ecis = copy.deepcopy(ecis)
    energy = []
    heat_cap = []
    not_computed_temps = []
    for i,T in enumerate(temps):
        print ("Current temperature {}K".format(T))
        if ( phonons ):
            try:
                with open("/home/davidkl/Documents/GPAWTutorial/CE/data/almg_eci_Fvib%d.json"%(T)) as infile:
                    phonon_ecis = json.load(infile)
            except Exception as exc:
                print (str(exc))
                not_computed_temps.append(T)
                continue
            # Remove higher than two terms
            ph = {}
            for key,value in phonon_ecis.items():
                if ( key.startswith("c2") ):
                    ph[key] = value
            phonon_ecis = ph
            corrected_ecis = update_phonons( copy.deepcopy(origin_ecis), phonon_ecis, T )
            print (corrected_ecis)
            ceBulk.atoms._calc.update_ecis(corrected_ecis)
        mc_obj = mc.Montecarlo( ceBulk.atoms, T )
        mc_obj.runMC( steps=n_burn, verbose=False )

        # Run Monte Carlo
        obs = mc_obs.PairCorrelationObserver( ceBulk.atoms._calc )
        mc_obj.attach( obs, interval=1 )
        mc_obj.runMC( steps=n_sampling )
        cfs.append( obs.get_average() )
        cf_std.append( obs.get_std() )
        n_samples.append( obs.n_entries )
        thermo = mc_obj.get_thermodynamic()
        energy.append( thermo["energy"] )
        heat_cap.append( thermo["heat_capacity"] )

    try:
        with open( out_file, 'r') as infile:
            data = json.load(infile)
    except:
        data = {}
        data["temperature"] = []
        data["cfs"] = []
        data["cf_std"] = []
        data["n_samples"] = []
        data["energy"] = []
        data["heat_capacity"] = []

    # Remove temperatures that ware not computed
    new_temps = []
    for T in temps:
        if ( T in not_computed_temps ):
            continue
        new_temps.append(T)
    temps = new_temps

    data["temperature"] += temps
    data["cfs"] += cfs
    data["cf_std"] += cf_std
    data["n_samples"] += n_samples
    data["energy"] += energy
    data["heat_capacity"] += heat_cap
    with open( out_file, 'w') as outfile:
        json.dump( data, outfile )

def main( run ):
    atoms = bulk("Al")
    atoms = atoms*(4,4,4)
    atoms[0].symbol = "Mg"

    db_name = "/home/davidkl/Documents/WangLandau/data/ce_hydrostatic_7x7.db"
    conc_args = {
        "conc_ratio_min_1":[[60,4]],
        "conc_ratio_max_1":[[64,0]],
    }
    ceBulk = CEBulk( "fcc", 4.05, None, [20,20,20], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, max_cluster_dia=1.414*4.05,reconf_db=False)
    init_cf = {key:1.0 for key in ecis.keys()}

    calc = CE( ceBulk, ecis, initial_cf=init_cf )
    ceBulk.atoms.set_calculator( calc )

    if ( run == "MC" ):
        mcmc( ceBulk, 0.1, phonons=False )
    else:
        chem_pot = {
        "Al":0.0,
        "Mg":0.0
        }
        gs_finder = SimmualtedAnnealingSGC( ceBulk.atoms, chem_pot, "test_db.db" )
        gs_finder.run( n_steps=1000, Tmin=400, ntemps=10 )
        gs_finder.show_visit_stat()
        gs_finder.show_compositions()
    view( ceBulk.atoms )

if __name__ == "__main__":
    #run = sys.argv[1] # Run is WL or MC
    run = "MC"
    main( run )
