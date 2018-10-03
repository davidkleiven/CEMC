from scipy.special import sph_harm
import numpy as np

class HarmonicsFit(object):
    """Class for fitting spherical harmonics to point cloud

    """
    def __init__(self, order=0):
        self.order = order
        self.coeff = None

    def __call__(self, u, v):
        """Evaluate the fit."""
        if self.coeff is None:
            raise ValueError("No coefficients have been fitted!")
        if isinstance(u, np.ndarray):
            res = np.zeros(u.shape)
        else:
            res = 0.0
        counter = 0
        for p in range(self.order+1):
            for m in range(-p, p+1):
                res += self.coeff[counter]*HarmonicsFit.real_spherical_harmonics(m, p, u, v)
                counter += 1
        return res

    @staticmethod
    def real_spherical_harmonics(m, p, u, v):
        """Return the real spherical harmonics."""
        if p < 0:
            return np.sqrt(2.0)*sph_harm(m, p, u, v).imag
        else:
            return np.sqrt(2.0)*sph_harm(m, p, u, v).real


    def fit(self, points, penalty=0.0):
        """Fit a sequence spherical harmonics to the data."""
        n = self.order
        num_terms = int(n + n*(n+1))
        num_terms = int((n+1)**2)
        A = np.zeros((points.shape[0], num_terms))
        col = 0
        for p in range(self.order+1):
            for m in range(-p, p+1):
                A[:, col] = HarmonicsFit.real_spherical_harmonics(m, p, points[:, 0], points[:, 1])
                col += 1
        
        N = A.shape[1]
        print(A.T.dot(A))
        matrix = np.linalg.inv(A.T.dot(A) + penalty*np.identity(N))
        self.coeff = matrix.dot(A.T.dot(points[:, 2]))

        pred = A.dot(self.coeff)
        rmse = np.sqrt(np.sum((pred-points[:, 2])**2)/len(pred))
        mean = np.mean(np.abs(points[:, 2]))
        print("RMSE harmonics fit: {}. Relative rmse: {}".format(rmse, rmse/mean))
        return self.coeff

    def show(self, n_angles=120):
        """Create a 3D visualization of the fitted shape.""" 
        from itertools import product
        from mayavi import mlab
        theta = np.linspace(0.0, np.pi, n_angles)
        phi = np.linspace(0.0, 2.0*np.pi, n_angles)
        theta = theta.tolist()
        T, P = np.meshgrid(theta, phi)
        radius = np.zeros(T.shape)
        print("Evaluating gamma at all angles...")
        radius = self(P, T)
        # for indx in product(range(n_angles), range(n_angles)):
        #    radius[indx] = self(P[indx], T[indx])

        X = radius*np.cos(P)*np.sin(T)
        Y = radius*np.sin(P)*np.sin(T)
        Z = radius*np.cos(T)
        mlab.mesh(X, Y, Z, scalars=radius)
        mlab.show()
