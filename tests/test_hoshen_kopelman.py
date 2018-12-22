import unittest
import numpy as np
try:
    from cemc_cpp_code import hoshen_kopelman
    available = True
    skip_msg = ""
except ImportError as exc:
    available = False
    skip_msg = str(exc)
    print(skip_msg)

class TestHoshenKopelman(unittest.TestCase):
    def test_simple_row(self):
        if not available:
            self.skipTest(skip_msg)
        data = np.zeros((3, 3, 3), dtype=np.uint8)
        data[:, :, 0] = 1
        clusters = hoshen_kopelman(data)
        self.assertTrue(np.all(clusters==data))

    def test_two_columns(self):
        if not available:
            self.skipTest(skip_msg)

        data = np.zeros((6, 6, 6), dtype=np.uint8)
        data[:2, :2, :2] = 1
        data[4, 4, :] = 1
        clusters = hoshen_kopelman(data)
        self.assertEqual(np.max(clusters), 2)
        expected_clusters = np.zeros((6, 6, 6), dtype=np.uint8)
        expected_clusters[:2, :2, :2] = 1
        expected_clusters[4, 4, :] = 2
        self.assertTrue(np.all(clusters == expected_clusters))

    def test_three_boxes(self):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[:2, :2, :2] = 1
        data[4:6, 4:6, 4:6] = 4
        data[8, 8, 8] = 5
        clusters = hoshen_kopelman(data)
        self.assertEqual(np.max(clusters), 3)
        expected_clusters = np.zeros((10, 10, 10), dtype=np.uint8)
        expected_clusters[:2, :2, :2] = 1
        expected_clusters[4:6, 4:6, 4:6] = 2
        expected_clusters[8, 8, 8] = 3
        self.assertTrue(np.all(clusters == expected_clusters))

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)