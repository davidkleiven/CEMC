import sys
sys.path.append( "/home/davidkl/Documents/WangLandau/wanglandau" )
from simple_ce_calc import CEcalc
from ase.build import bulk
from ase.ce.settings import BulkCrystal
from sa_sgc import SimmualtedAnnealingSGC
from ase.visualize import view

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

def main():
    atoms = bulk("Al")
    atoms = atoms*(4,4,4)
    atoms[0].symbol = "Mg"

    db_name = "/home/davidkl/Documents/GPAWTutorial/CE/ce_hydrostatic.db"
    conc_args = {
        "conc_ratio_min_1":[[60,4]],
        "conc_ratio_max_1":[[64,0]],
    }
    ceBulk = BulkCrystal( "fcc", 4.05, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)

    calc = CEcalc( ecis, ceBulk )
    atoms.set_calculator( calc )

    chem_pot = {
    "Al":0.0,
    "Mg":0.0
    }
    gs_finder = SimmualtedAnnealingSGC( atoms, chem_pot, "test_db.db" )
    gs_finder.run()
    view( atoms )

if __name__ == "__main__":
    main()
