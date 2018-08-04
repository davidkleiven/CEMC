"""Useful functions when using the dataset package."""
class RequiredFieldMissingError(Exception):
    pass

def atoms2dict(atoms):
    """Dictionary representation of atoms object."""
    info = {
        "cell": atoms.get_cell(),
        "positions": atoms.get_positions(),
        "pbc": atoms.get_pbc(),
        "numbers": atoms.get_atomic_numbers()
    }
    return info


def dict2atoms(info):
    """Extract quantities from and construct an atoms object."""
    required_fields = ["positions", "cell", "pbc", "numbers"]
    for field in required_fields:
        if field not in info.keys():
            msg = "To convert to atoms object the following fields are "
            msg += "mandatory: {}".format(required_fields)
            raise RequiredFieldMissingError(msg)

    from ase import Atoms
    atoms = Atoms(positions=info["positions"],
                  pbc=info["pbc"],
                  cell=info["cell"],
                  numbers=info["numbers"])
    return atoms
