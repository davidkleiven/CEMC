import numpy as np
import json


def cahn_hilliard_surface_parameter(conc, free_energy, interfacial_energy):
    """Calculate the gradient parameter according to Cahn-Hilliard

        Free Energy of a Nonuniform System. I. Interfacial Free Energy
        Cahn, John W. "JW Cahn and JE Hilliard,
        J. Chem. Phys. 28, 258 (1958)."
        J. Chem. Phys. 28 (1958): 258.

    :param np.ndarray conc: Concentrations
    :param np.ndarray free_energy: Free energy difference
    :param float interfacial_energy: Interfacial energy in the same energy units
    """

    integral = np.trapz(np.sqrt(free_energy), x=conc)
    return (0.5*interfacial_energy/integral)**2


def get_polyterms(fname):
    """Parse JSON file and return list of PyPolyterms.

    :param str fname: JSON file with the parameters
    """
    from phasefield_cxx import PyPolynomialTerm

    with open(fname, 'r') as infile:
        data = json.load(infile)

    poly_terms = []
    coefficients = []
    for entry in data["terms"]:
        poly_terms.append(PyPolynomialTerm(entry["powers"]))
        coefficients.append(entry["coeff"])
    return coefficients, poly_terms
