import unittest
try:
    from cemc.tools import BinaryCriticalPoints
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(reason)


class TestBinaryCriticalPoints(unittest.TestCase):
    def test_simple(self):
        if not available:
            self.skipTest(reason)

        bc = BinaryCriticalPoints()

        c1 = 0.1
        c2 = 0.9
        p1 = [0.7, -2*0.7*c1, 0.7*c1**2]
        p2 = [0.9, -2*0.9*c2, 0.9*c2**2]
        x1, x2 = bc.coexistence_points(p1, p2)
        self.assertAlmostEqual(x1, c1)
        self.assertAlmostEqual(x2, c2)

    def test_spinodal(self):
        if not available:
            self.skipTest()
            bc = BinaryCriticalPoints()

            poly = [1, -3, 5, 6]

            # Just make sure that the functions run
            bc.spinodal(poly)
            bc.plot([1, 2], [2, 1], polys=[poly])

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
