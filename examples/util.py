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
    names = bc.cluster_family_names
    for name in names:
        if int(name[1]) == 2:
            return name
    raise RuntimeError("No pair cluster found!")
