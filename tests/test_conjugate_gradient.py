import unittest
import numpy as np
try:
    from phasefield_cxx import PySparseMatrix, PyConjugateGradient
    available = True
    reason = ""
except ImportError as exc:
    reason = str(exc)
    available = False

class TestConjugateGradient(unittest.TestCase):
    def test_sp_mat_symmetric(self):
        if not available:
            self.skipTest(reason)

        matrix = [[1.0, 2.0, 0.0, 4.5],
                  [-2.4, 0.1, 0.2, 3.2],
                  [0.7, -0.2, 1.0, 0.0],
                  [0.0, -2.4, 2.5, 0.7]
                  ]
        matrix = np.array(matrix)

        sp_matrix = PySparseMatrix()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                sp_matrix.insert(i, j, matrix[i, j])
        
        self.assertFalse(sp_matrix.is_symmetric())

        matrix = matrix.T + matrix
        sp_matrix.clear()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                sp_matrix.insert(i, j, matrix[i, j])
        
        self.assertTrue(sp_matrix.is_symmetric())
        
    def test_solve(self):
        sp_matrix = PySparseMatrix()
        cg = PyConjugateGradient(1E-4)

        matrix = [[1.0, 2.0, 0.0, 4.5],
                  [-2.4, 0.1, 0.2, 3.2],
                  [0.7, -0.2, 1.0, 0.0],
                  [0.0, -2.4, 2.5, 0.7]
                  ]
        matrix = np.array(matrix)

        matrix = 0.5*(matrix.T + matrix)
        rhs = np.array([0.4, 0.1, -0.4, 0.9])
        x = np.linalg.solve(matrix, rhs)

        dot1 = matrix.dot(rhs)

        # Fill the sparse matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                sp_matrix.insert(i, j, matrix[i, j])

        dot2 = sp_matrix.dot(rhs)
        self.assertTrue(np.allclose(dot1, dot2))
        
        x0 = np.zeros(4)
        #sp_matrix.save("sparse_mat.csv")

        x_cg = cg.solve(sp_matrix, rhs, x0)

        #self.assertTrue(np.allclose(x, x_cg))

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
            