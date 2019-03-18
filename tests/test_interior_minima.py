import unittest
import numpy as np

try:
    from cemc.phasefield import InteriorMinima
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)


class TestInteriorMinima(unittest.TestCase):
    def test_minima(self):
        x = list(range(0, 20))
        X, Y = np.meshgrid(x, x)

        data = ((16 - Y**2)*(9 - X**2))**2
        data = ((X-3)**2 + (Y-4)**2)**2
        int_min = InteriorMinima()
        minima = int_min.find_extrema(data)
        self.assertEqual(set(minima), set([(4, 3)]))


def show_data(data):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, vmin=0, vmax=2)
    plt.show()

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)