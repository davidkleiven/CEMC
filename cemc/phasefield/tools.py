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


def get_polyterms(fname):
    """Parse csv file and return list of PyPolyterms.

    :param str fname: CSV file with coefficients and powers
    """
    from phasefield_cxx import PyPolynomialTerm

    data = np.loadtxt(fname, delimiter=",")

    poly_terms = []
    coefficients = []
    for row in range(data.shape[0]):
        coeff = data[row, 0]
        inner_pow = data[row, 1:-1].astype(np.int32)
        outer_pow = int(data[row, -1])

        poly_terms.append(PyPolynomialTerm(inner_pow, outer_pow))
        coefficients.append(coeff)
    return coefficients, poly_terms