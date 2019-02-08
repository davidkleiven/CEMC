import unittest
import numpy as np

try:
    from cemc_cpp_code import PyMat4D
    available = True
    reason = ""
except ImportError as exc:
    available = False
    reason = str(exc)

print(reason)

class TestMat4D(unittest.TestCase):
    def test_numpy(self):
        if not available:
            self.skipTest(reason)

        array = np.random.rand(3, 3, 3, 3).astype(np.float64)
        mat4D = PyMat4D()
        mat4D.from_numpy(array)
        # ret_array = mat4D.to_numpy()
        # self.assertTrue(np.allclose(array, ret_array))

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)