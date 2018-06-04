from ase.ce import CorrFunction
from ase.ce import BulkCrystal

def get_example_ecis(bc=None,bc_kwargs=None):
    if ( bc is not None ):
        cf = CorrFunction(bc)
    else:
        bc = BulkCrystal(**bc_kwargs)
        cf = CorrFunction(bc)
    cf = cf.get_cf(bc.atoms)
    eci = {key:0.001 for key in cf.keys()}
    return eci

def get_example_network_name(bc):
    return bc.cluster_names[0][2][0]
