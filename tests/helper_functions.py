from ase.ce import BulkSpacegroup, BulkCrystal
from cemc.wanglandau.ce_calculator import CE, get_ce_calc

def get_bulkspacegroup_binary():
# https://materials.springer.com/isp/crystallographic/docs/sd_0453869
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Al","Al","Al","Al"]
    #symbols = ["Al","Al","Al","Al"]
    basis = [(0,0,0),(0.324,0.324,0.324),(0.3582,0.3582,0.0393),(0.0954,0.0954,0.2725)]
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]]
    }
    db_name = "test_db_binary_almg.db"
    basis_elements = [["Al","Mg"],["Al","Mg"],["Al","Mg"],["Al","Mg"]]
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=217, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=4, db_name=db_name, size=[1,1,1], grouped_basis=[[0,1,2,3]] )
    return bs, db_name



def get_small_BC_with_ce_calc(lat="fcc"):
    db_name = "test_db_{}.db".format(lat)

    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]],
    }
    a = 4.05
    ceBulk = BulkCrystal( crystalstructure=lat, a=a, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, \
    db_name=db_name, max_cluster_size=4)
    ceBulk.reconfigure_settings()
    ceBulk._get_cluster_information()
    cf = CorrFunction(ceBulk)
    corrfuncs = cf.get_cf(ceBulk.atoms)
    eci = {name:1.0 for name in corrfuncs.keys()}
    calc = CE( ceBulk, eci )
    ceBulk.atoms.set_calculator(calc)
    return ceBulk
