import unittest
from ase.io import read
try:
    from cemc.tools import WulffConstruction
    available = True
except ImportError as exc:
    io_msg = str(exc)
    available = False


class TestWulff(unittest.TestCase):
    def test_no_throw(self):
        if not available:
            self.skipTest(io_msg)

        try:
            fname = "tests/test_data/cluster.xyz"
            atoms = read(fname)
        except IOError as exc:
            msg = str("IOError_{}".format(str(exc)))
            self.skipTest(msg)

        msg = ""
        no_throw = True
        try:
            wulff = WulffConstruction(atoms, max_dist_in_element=6.0)
            wulff.surface_atoms
            wulff.interface_energy
            wulff.interface_energy_poly_expansion(order=2, spg=225, show=True)
        except Exception as exc:
            no_throw = False
            msg = "{}_{}".format(type(exc).__name__, str(exc))
        self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
