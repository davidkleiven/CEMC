import numpy as np


def fit_kernel(x=[], y=[], num_kernels=1, kernel=None, lamb=None,
               extrapolate="none", extrap_range=0.1):
    from phasefield_cxx import PyKernelRegressor

    allowed_extrapolations = ["none", "linear"]
    if extrapolate not in allowed_extrapolations:
        raise ValueError("extrapolate has to be one of {}"
                         "".format(allowed_extrapolations))

    if extrapolate == "linear":
        # Add extrapolation to the beginning
        extrap_range = extrap_range*(x[-1] - x[0])
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        slope = dy/dx
        x_extrap = np.arange(x[0] - extrap_range, x[0], dx)
        y_extrap = y[0] + (dy/dx)*(x_extrap - x[0])
        x = np.concatenate((x_extrap, x))
        y = np.concatenate((y_extrap, y))

        # Add extrapolation to end
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        x_extrap = np.arange(x[-1], x[-1] + extrap_range, dx)
        y_extrap = y[-1] + (dy/dx)*(x_extrap - x[-1])
        x = np.concatenate((x, x_extrap))
        y = np.concatenate((y, y_extrap))

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


def heaviside(x):
    return float(x > 0.0)


def smeared_heaviside(x, w):
    return (1 + np.tanh(x/w))
