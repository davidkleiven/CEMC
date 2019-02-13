import numpy as np


def cahn_hilliard_surface_parameter(conc, free_energy, interfacial_energy, density):
    """Calculate the gradient parameter according to Cahn-Hilliard

        Free Energy of a Nonuniform System. I. Interfacial Free Energy
        Cahn, John W. "JW Cahn and JE Hilliard,
        J. Chem. Phys. 28, 258 (1958)."
        J. Chem. Phys. 28 (1958): 258.

    :param np.ndarray conc: Concentrations
    :param np.ndarray free_energy: Free energy difference
    :param float interfacial_energy: Interfacial energy in the same energy units
    :param float density: Number density
    """

    integral = np.trapz(np.sqrt(free_energy), x=conc)
    return (0.5*interfacial_energy/(density*integral))**2
