import numpy as np


def fit_kernel(x=[], y=[], num_kernels=1, kernel=None, lamb=None):
    from phasefield_cxx import PyKernelRegressor
    regressor = PyKernelRegressor(np.min(x), np.max(x))

    coeff = np.zeros(num_kernels)
    regressor.set_kernel(kernel)
    regressor.set_coeff(coeff)

    matrix = np.zeros((len(x), num_kernels))

    if num_kernels >= len(x):
        raise ValueError("The number of kernels has to be lower than "
                         "the number points!")

    for i in range(num_kernels):
        matrix[:, i] = regressor.evaluate_kernel(i, x)

    if lamb is None:
        coeff = np.linalg.lstsq(matrix, y)[0]
    else:
        u, d, vh = np.linalg.svd(matrix, full_matrices=False)
        D = np.diag(d/(lamb + d**2))
        Uy = u.T.dot(y)
        DUy = D.dot(Uy)
        coeff = vh.T.dot(DUy)
    regressor.set_coeff(coeff)
    return regressor

