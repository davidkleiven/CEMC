import unittest
try:
    import numpy as np
    from cemc.tools.phasediagram import SingleCurvaturePolynomial
    available = True
    reason = ""
except Exception as exc:
    available = False
    reason = str(exc)


class TestSingleCurvaturePoly(unittest.TestCase):
    def test_parabola(self):
        if not available:
            self.skipTest(reason)
        x = np.linspace(0.0, 10.0, 40)

        a = 1.0
        b = 2.0
        c = 3.0
        y = a*x**2 + b*x + c

        single = SingleCurvaturePolynomial(curvature="convex")
        coeff = single.fit(x, y)

        self.assertTrue(np.allclose(coeff, [a, b, c]))

    def test_concave_fit_to_convex_curve(self):
        if not available:
            self.skipTest(reason)
        x = np.linspace(0.0, 10.0, 40)

        a = 1.0
        b = 2.0
        c = 3.0
        y = a*x**2 + b*x + c

        single = SingleCurvaturePolynomial(curvature="concave")
        coeff = single.fit(x, y)
        self.assertLessEqual(coeff[0], 0.0)

    def test_decreasing(self):
        if not available:
            self.skipTest(reason)
        x = np.linspace(0.0, 10.0, 40)

        a = 1.0
        b = 2.0
        y = a*x + b
        single = SingleCurvaturePolynomial(slope="increasing")
        coeff = single.fit(x, y, order=1)
        self.assertTrue(np.allclose(coeff, [a, b]))

    def test_decreasing_to_increasing_poly(self):
        if not available:
            self.skipTest(reason)
        x = np.linspace(0.0, 10.0, 40)

        a = 1.0
        b = 2.0
        y = a*x + b
        single = SingleCurvaturePolynomial(slope="decreasing", alpha=1E6)
        coeff = single.fit(x, y, order=1)
        self.assertAlmostEqual(coeff[0], 0.0, places=4)
        print(coeff)
        self.assertAlmostEqual(coeff[1], np.mean(y), places=0)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)