from cemc.ce_calculator import CE, get_ce_calc
from ase.ce import BulkSpacegroup
from ase.ce import BulkCrystal
from inspect import getargspec
from ase.ce import CorrFunction

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
    max_dia = get_max_cluster_dia_name()
    size_arg = {max_dia:4.1}
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=217, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=3, db_name=db_name, size=[1, 1, 1], grouped_basis=[[0,1,2,3]], **size_arg )
    return bs, db_name


def get_small_BC_with_ce_calc(lat="fcc"):
    db_name = "test_db_{}.db".format(lat)

    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]],
    }
    a = 4.05
    ceBulk = BulkCrystal( crystalstructure=lat, a=a, size=[3,3,3], basis_elements=[["Al","Mg"]], conc_args=conc_args, \
    db_name=db_name, max_cluster_size=3)
    ceBulk.reconfigure_settings()
    cf = CorrFunction(ceBulk)
    corrfuncs = cf.get_cf(ceBulk.atoms)
    eci = {name:1.0 for name in corrfuncs.keys()}
    calc = CE( ceBulk, eci )
    ceBulk.atoms.set_calculator(calc)
    return ceBulk

def get_ternary_BC():
    db_name = "test_db_ternary.db"
    conc_args = {
        "conc_ratio_min_1":[[4,0,0]],
        "conc_ratio_max_1":[[0,4,0]],
        "conc_ratio_min_1":[[2,2,0]],
        "conc_ratio_max_2":[[1,1,2]]
    }
    max_dia = get_max_cluster_dia_name()
    size_arg = {max_dia:4.05}
    ceBulk = BulkCrystal(crystalstructure="fcc", a=4.05, size=[4,4,4], basis_elements=[["Al","Mg","Si"]], \
                         conc_args=conc_args, db_name=db_name, max_cluster_size=3, **size_arg)
    ceBulk.reconfigure_settings()
    return ceBulk

def get_max_cluster_dia_name():
    """
    In former versions max_cluster_dist was called max_cluster_dia
    """
    kwargs = {"max_cluster_dia":5.0}
    argspec = getargspec(BulkCrystal.__init__).args
    if "max_cluster_dia" in argspec:
        return "max_cluster_dia"
    return "max_cluster_dist"

def flatten_cluster_names(cnames):
    flattened = []
    for sub in cnames:
        for sub2 in sub:
            flattened += sub2
    return flattened

def get_example_network_name(bc):
    names = bc.cluster_family_names
    for name in names:
        if int(name[1]) == 2:
            return name
    raise RuntimeError("No pair name was found!")

def get_example_ecis(bc=None):
    cf = CorrFunction(bc)
    cf = cf.get_cf(bc.atoms)
    eci = {key:0.001 for key in cf.keys()}
    return eci

def get_example_cf(bc=None):
    cf = CorrFunction(bc)
    cf = cf.get_cf(bc.atoms)
    return cf
