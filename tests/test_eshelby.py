"""Unittest for the Eshelby tensor."""
import unittest
from cemc.ce_updater import EshelbyTensor, EshelbySphere
from itertools import product


class TestEshelby(unittest.TestCase):
    """Class for testing the Eshelby tensor"""

    def test_minor_symmetry_tensor(self):
        """Test that the Eshelby tensor exhibt minor symmetry."""
        e_tensor = EshelbyTensor(6.0, 5.0, 4.0, 0.3)

        for indx in product([0, 1, 2], repeat=4):
            val1 = e_tensor(indx[0], indx[1], indx[2], indx[3])

            val2 = e_tensor(indx[0], indx[1], indx[3], indx[2])
            self.assertAlmostEqual(val1, val2)

            val2 = e_tensor(indx[1], indx[0], indx[3], indx[2])
            self.assertAlmostEqual(val1, val2)

            val2 = e_tensor(indx[1], indx[0], indx[2], indx[3])
            self.assertAlmostEqual(val1, val2)


if __name__ == "__main__":
    unittest.main()
