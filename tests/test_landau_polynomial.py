import unittest
import numpy as np
import os
try:
    from cemc.tools import TwoPhaseLandauPolynomial
    from phasefield_cxx import PyTwoPhaseLandau
    from cemc.phasefield.tools import get_polyterms
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)
    print(str(exc))


class TestLandauPolynomial(unittest.TestCase):
    def test_fit(self):
        if not available:
            self.skipTest(reason)
        error = False
        msg = ""
        try:
            conc1 = np.linspace(0.0, 0.3, 100)
            F1 = conc1**2
            conc2 = np.linspace(0.6, 1.0, 100)
            F2 = 0.5*(conc2-0.9)**2
            F = np.concatenate((F1, F2))
            conc = np.concatenate((conc1, conc2))
            poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5)
            poly.fit(conc, F)
            # show_fit(poly, conc1, conc2, F1, F2)
        except Exception as exc:
            error = True
            msg = str(exc)

        self.assertFalse(error, msg=msg)

    def test_Cxx_poly(self):
        if not available:
            self.skipTest(reason)
        c1 = 0.0
        c2 = 0.8
        coeff = [1.0, 2.0, 3.0, 4.0]
        pypoly = PyTwoPhaseLandau(c1, c2, coeff)

        shape = [1.0, 0.0, 0.0]
        shape_npy = np.array(shape)
        c = 0.5
        expect = coeff[0]*(c-c1)**2 + \
            coeff[1]*(c - c2)*np.sum(shape_npy**2) + \
            coeff[2]*np.sum(shape_npy**4) + coeff[3]*np.sum(shape_npy**2)**3
        self.assertAlmostEqual(pypoly.evaluate(c, shape), expect)

    def test_partial_derivative(self):
        if not available:
            self.skipTest(reason)
        
        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5)
        poly.conc_coeff = [1, -2, 3]
        poly.conc_coeff2 = [-1, 2, 0]
        poly.coeff_shape[0] = 2
        poly.coeff_shape[2] = 5

        # Verify that we get a value error if we pass an unknown
        # variable
        with self.assertRaises(ValueError):
            poly.partial_derivative(0.5, shape=0.5, var="var2")

        conc = 0.4
        shape = 0.2
        pd_conc = poly.partial_derivative(conc, shape=shape, var="conc")
        expected = np.polyval(np.polyder(poly.conc_coeff), conc)
        expected += np.polyval(np.polyder(poly.conc_coeff2), conc)*shape**2
        self.assertAlmostEqual(pd_conc, expected)

        pd_shape = poly.partial_derivative(conc, shape=shape, var="shape")
        expected = -0.0104
        expected = np.polyval(poly.conc_coeff2, conc)*2*shape + \
            4*2*shape**3 + 6*5*shape**5

        self.assertAlmostEqual(pd_shape, expected)

    def test_equil_deriv(self):
        if not available:
            self.skipTest(reason)
        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5)
        poly.conc_coeff = [1, -2, 3]
        poly.conc_coeff2 = [-1, 3, -1.5]
        poly.coeff_shape[0] = 2
        poly.coeff_shape[2] = 5

        conc = 0.4
        deriv = poly.equil_shape_order_derivative(conc)
        self.assertGreater(abs(deriv), 0.0)

        delta = 0.00001
        val1 = poly.equil_shape_order(conc)
        val2 = poly.equil_shape_order(conc + delta)
        fd = (val2 - val1)/delta
        self.assertAlmostEqual(fd, deriv, places=4)

    def test_equil_fixed_shape(self):
        if not available:
            self.skipTest(reason)
        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5, conc_order1=2,
                                        conc_order2=1)
        conc_coeff = [1.0, 0.0, 0.0]
        conc_coeff1 = [2.0, 0.0]
        poly.coeff_shape[0] = -2.0
        poly.coeff_shape[2] = 5.0
        n_eq = poly.equil_shape_order(0.5)
        self.assertGreater(n_eq, 0.0)

        poly.coeff_shape[1] = 2.0
        n_eq2 = poly.equil_shape_fixed_conc_and_shape(0.5, shape=n_eq)
        self.assertAlmostEqual(n_eq2, 0.0)

        n_eq2 = poly.equil_shape_fixed_conc_and_shape(0.5, shape=0.0)
        self.assertAlmostEqual(n_eq2, n_eq)

        n_eq2 = poly.equil_shape_fixed_conc_and_shape(0.5, shape=n_eq,
                                                      min_type="mixed")
        self.assertAlmostEqual(n_eq2, 0.0)

        n_eq2 = poly.equil_shape_fixed_conc_and_shape(0.5, shape=0.0,
                                                      min_type="mixed")
        self.assertLess(n_eq2, n_eq)

    def test_equil_fixed_shape_deriv(self):
        if not available:
            self.skipTest(reason)

        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5, conc_order1=2,
                                        conc_order2=1)
        conc_coeff = [1.0, 0.0, 0.0]
        conc_coeff1 = [2.0, 0.0]
        poly.coeff_shape[0] = -2.0
        poly.coeff_shape[2] = 5.0
        poly.coeff_shape[1] = 2.0
        poly.coeff_shape[3] = -3.0
        poly.coeff_shape[4] = 6.0
        n_eq = poly.equil_shape_order(0.5)
        self.assertGreater(n_eq, 0.0)

        conc = 0.5
        n_eq_eval = 0.3
        n_eq2 = poly.equil_shape_fixed_conc_and_shape(conc, shape=n_eq_eval)

        self.assertGreater(n_eq2, 0.0)
        deriv = poly.equil_shape_fixed_conc_and_shape_deriv(
            conc, shape=n_eq_eval, min_type="pure")

        delta = 1E-4
        n_eq2_delta = poly.equil_shape_fixed_conc_and_shape(
            conc, shape=n_eq_eval+delta)
        self.assertGreater(n_eq2_delta, 0.0)

        fd_deriv = (n_eq2_delta - n_eq2)/delta
        self.assertAlmostEqual(fd_deriv, deriv, places=4)

        # Change the coefficient to construct an equillibrium position
        # with mixed layers
        poly.coeff_shape[3] = -8.0
        delta = 1E-5
        n_eq2 = poly.equil_shape_fixed_conc_and_shape(conc, shape=n_eq_eval,
                                                      min_type="mixed")
        self.assertGreater(n_eq2, 0.0)

        deriv = poly.equil_shape_fixed_conc_and_shape_deriv(
            conc, shape=n_eq_eval, min_type="mixed")

        n_eq2_delta = poly.equil_shape_fixed_conc_and_shape(
            conc, shape=n_eq_eval+delta, min_type="mixed")
        self.assertGreater(n_eq2_delta, 0.0)

        fd_deriv = (n_eq2_delta - n_eq2)/delta
        self.assertAlmostEqual(fd_deriv, deriv, places=4)

    def test_array_decorator_only_conc(self):
        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5, conc_order1=2,
                                        conc_order2=1)

        # Case 1: Concentration is a scalar
        poly.eval_at_equil(0.5)

        # Case 2: Concentration is a list
        poly.eval_at_equil([0.5, 0.1, 0.2])

        # Case 3: Concentration is a Numpy array
        poly.eval_at_equil(np.array([0.5, 0.1, 0.2]))

        # Case 4: Concentration is s 2D Numpy array
        with self.assertRaises(ValueError):
            poly.eval_at_equil(np.array([[0.5, 0.2], [0.1, 0.2]]))

        # Case 5: Concentration is a nested list
        with self.assertRaises(ValueError):
            poly.eval_at_equil([[0.5, 0.2], [0.1, 0.2]])

    def test_array_decorator_mixed(self):
        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.5, conc_order1=2,
                                        conc_order2=1)

        # Case 1: Concentration is a scalar, shape is None
        poly.evaluate(0.5, shape=None)

        # Case 2: Concentration is a list, shape is None
        poly.evaluate([0.5, 0.2], shape=None)

        # Case 3: Concentration is a Numpy array, shape is None
        poly.evaluate(np.array([0.5, 0.2]), shape=None)

        # Case 4: Concentration is nested list shape is None
        with self.assertRaises(ValueError):
            poly.evaluate([[0.5, 0.2], [0.2, 0.1]], shape=None)

        # Case 5: Concentration is a 2D Numpy array shape is None
        with self.assertRaises(ValueError):
            poly.evaluate(np.array([[0.5, 0.2], [0.2, 0.1]]), shape=None)

        # Case 6: Concentration scalar, shape scalar
        poly.evaluate(0.5, shape=0.2)

        # Case 7: Concentration scalar, shape 1D list
        poly.evaluate(0.5, shape=[0.1, 0.2, 0.4])

        # Case 8: Concentration scalar, shape 1D list of wrong length
        with self.assertRaises(ValueError):
            poly.evaluate(0.5, shape=[0.1, 0.2])

        # Case 9: Concentration list, shape list
        poly.evaluate([0.2, 0.3], shape=[[0.1, 0.0, 0.0],
                                         [0.2, 0.3, 0.4]])

        # Case 10: Concentration list and shape list has wrong dimensions
        with self.assertRaises(ValueError):
            poly.evaluate([0.2, 0.3], shape=[[0.1, 0.0, 0.0]])

    def test_export(self):
        poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.0, conc_order1=3,
                                        conc_order2=4)
        poly.conc_coeff[:] = np.array([4.0, 5.0, -2.0, 1.0])
        poly.conc_coeff2[:] = np.array([2.0, -1.0, 0.2, -5.0, 10.0])
        poly.coeff_shape[:] = [2.0, -1.0, 3.0, 2.3, 5.0]

        fname = "landau_export.json"
        poly.save_poly_terms(fname=fname)

        # Get the polynomials
        try:
            coefficients, poly_terms = get_polyterms(fname)
        except ImportError as exc:
            os.remove(fname)
            self.skipTest(str(exc))

        # Construct a polynomial from this
        poly_raw = None
        try:
            from phasefield_cxx import PyPolynomial
            poly_raw = PyPolynomial(4)
        except ImportError as exc:
            os.remove(fname)
            self.skipTest(str(exc))

        for c, term in zip(coefficients, poly_terms):
            poly_raw.add_term(c, term)

        # Try to evaluate
        conc = 0.5
        shape = [0.2, 0.6, 0.1]
        expect = poly.evaluate(conc, shape=shape)
        exported_value = poly_raw.evaluate([conc] + shape)
        self.assertAlmostEqual(expect, exported_value)
        os.remove(fname)



def show_fit(poly, conc1, conc2, F1, F2):
    import matplotlib as mpl
    mpl.rcParams.update({"svg.fonttype": "none",
                         "font.size": 18,
                         "axes.unicode_minus": False})

    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(conc1, F1, "x")
    ax.plot(conc2, F2, "o", mfc="none")
    c = np.linspace(np.min(conc1), np.max(conc2), 100)
    fitted = poly.eval_at_equil(c)
    ax.plot(c, fitted)

    ax2 = ax.twinx()
    n_eq = poly.equil_shape_order(c)
    ax2.plot(c, n_eq)
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Free energy")
    ax2.set_ylabel("Long range order parameter")
    plt.show()




if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
