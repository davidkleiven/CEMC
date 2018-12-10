"""Unit tests for the strain energy."""
try:
    import unittest
    import numpy as np
    from cemc.tools import StrainEnergy
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)

C_al = np.array([[0.62639459, 0.41086487, 0.41086487, 0, 0, 0],
                [0.41086487, 0.62639459, 0.41086487, 0, 0, 0],
                [0.41086487, 0.41086487, 0.62639459, 0, 0, 0],
                [0, 0, 0, 0.42750351, 0, 0],
                [0, 0, 0, 0, 0.42750351, 0],
                [0, 0, 0, 0, 0, 0.42750351]])


def kato_et_al_sphere(f, mu, misfit):
    """Return the equivalent strain for a sphere.

    Kato, M.; Fujii, T. & Onaka, S.
    Elastic strain energies of sphere, plate and needle inclusions
    Materials Science and Engineering: A,
    Elsevier, 1996, 211, 95-103
    """
    eq_strain = np.zeros(6)

    # exx
    f1 = (10*misfit[0] - 5*(misfit[1]+misfit[2]))/(2*f*(4-5*mu) + (7-5*mu))
    f2 = (misfit[0] + misfit[1] + misfit[2])/(f*(1+mu) + 2*(1-2*mu))
    eq_strain[0] = f*(1-mu)*(f1+f2)

    # eyy
    f1 = (10*misfit[1] - 5*(misfit[2]+misfit[0]))/(2*f*(4-5*mu) + (7-5*mu))
    f2 = (misfit[0] + misfit[1] + misfit[2])/(f*(1+mu) + 2*(1-2*mu))
    eq_strain[1] = f*(1-mu)*(f1+f2)

    # ezz
    f1 = (10*misfit[2] - 5*(misfit[0]+misfit[1]))/(2*f*(4-5*mu) + (7-5*mu))
    f2 = (misfit[0] + misfit[1] + misfit[2])/(f*(1+mu) + 2*(1-2*mu))
    eq_strain[2] = f*(1-mu)*(f1+f2)

    # Shear components
    eq_strain[5] = 15*f*(1-mu)*misfit[5]/(2*f*(4-5*mu) + (7-5*mu))
    eq_strain[4] = 15*f*(1-mu)*misfit[4]/(2*f*(4-5*mu) + (7-5*mu))
    eq_strain[3] = 15*f*(1-mu)*misfit[3]/(2*f*(4-5*mu) + (7-5*mu))
    return eq_strain


def kato_et_al_plate(f, mu, misfit):
    """Return the equivalent strain for a plate.

    Kato, M.; Fujii, T. & Onaka, S.
    Elastic strain energies of sphere, plate and needle inclusions
    Materials Science and Engineering: A,
    Elsevier, 1996, 211, 95-103
    """
    eq_strain = np.zeros(6)
    eq_strain[0] = f*misfit[0]
    eq_strain[1] = f*misfit[1]
    eq_strain[2] = (1-f)*mu*(misfit[0] + misfit[1])/(1-mu) + misfit[2]
    eq_strain[5] = f*misfit[5]
    eq_strain[3] = misfit[3]
    eq_strain[4] = misfit[4]
    return eq_strain


def kato_et_al_needle(f, mu, misfit):
    """Return the equivalent strains for a needle."""
    eq_strain = np.zeros(6)

    # e_zz
    f1 = (f*(5-4*mu) + (3-4*mu))*misfit[1]
    f2 = (f-1)*(1-4*mu)*misfit[2]
    div = (f*(3-4*mu) + 1)*(f + (1-2*mu))
    f3 = f*(1-f)*mu*misfit[0]/(f + 1-2*mu)
    eq_strain[1] = f*(1-mu)*(f1 + f2)/div + f3

    # e_yy
    f1 = (f*(5-4*mu) + (3-4*mu))*misfit[2]
    f2 = (f-1)*(1-4*mu)*misfit[1]
    div = (f*(3-4*mu) + 1)*(f + (1-2*mu))
    f3 = f*(1-f)*mu*misfit[0]/(f + 1-2*mu)
    eq_strain[2] = f*(1-mu)*(f1 + f2)/div + f3

    # e_xx
    eq_strain[0] = f*misfit[0]

    #_e_yz
    eq_strain[3] = 4*f*(1-mu)*misfit[3]/(f*(3-4*mu) + 1)

    # e_zx
    eq_strain[4] = 2*f*misfit[4]/(1+f)

    # e_yx
    eq_strain[5] = 2*f*misfit[5]/(1+f)
    return eq_strain


class TestStrainEnergy(unittest.TestCase):
    """Unit tests for strain energy class."""

    def test_equiv_strain_sphere(self):
        """Test the equivalent eigenstrain against analytic results.

        Kato, M.; Fujii, T. & Onaka, S.
        Elastic strain energies of sphere, plate and needle inclusions
        Materials Science and Engineering: A,
        Elsevier, 1996, 211, 95-103
        """
        if not available:
            self.skipTest(reason)
        mu = 0.3
        misfit = [0.05, 0.04, 0.03, 0.03, 0.02, 0.01]
        strain_eng = StrainEnergy(aspect=[2.0, 2.0, 2.0],
                                  misfit=misfit, poisson=mu)
        f = 5.0
        eq_strain = strain_eng.equivalent_eigenstrain(C_matrix=C_al, scale_factor=f)
        kato_et_al = kato_et_al_sphere(f, mu, misfit)
        self.assertTrue(np.allclose(eq_strain, kato_et_al, atol=1E-5))

    def test_equiv_strain_plate(self):
        """Test the equivalent strains for a plate."""
        if not available:
            self.skipTest(reason)
        mu = 0.3
        misfit = [0.05, 0.04, 0.03, 0.03, 0.02, 0.01]
        strain_eng = StrainEnergy(aspect=[200000000.0, 51200000.0, 0.001],
                                  misfit=misfit, poisson=mu)
        f = 4.0
        eq_strain = strain_eng.equivalent_eigenstrain(C_matrix=C_al, scale_factor=f)
        kato = kato_et_al_plate(f, mu, misfit)
        self.assertTrue(np.allclose(eq_strain, kato, atol=1E-3))

    def test_equiv_strain_needle(self):
        """Test the equivalent strains for a needle."""
        if not available:
            self.skipTest(reason)
        mu = 0.3
        misfit = [0.05, 0.04, 0.03, 0.03, 0.02, 0.01]
        strain_eng = StrainEnergy(aspect=[50000.0, 5.0, 5.0],
                                  misfit=misfit, poisson=mu)
        f = 9.0
        eq_strain = strain_eng.equivalent_eigenstrain(C_matrix=C_al, scale_factor=f)
        kato = kato_et_al_needle(f, mu, misfit)
        self.assertTrue(np.allclose(eq_strain, kato, atol=1E-3))


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
