class Involution(object):
    def __call__(self, x):
        raise NotImplementedError("Call method has to be implemented in child class!")

    def __deriv__(self, x):
        raise NotImplementedError("Deriv method has to be implemented in child class!")

    def _check_valid(self, x, xmin, xmax):
        if x < xmin or x > xmax:
            raise ValueError("{} < x < {} for involution".format(xmin, xmax))


class LinearInvolution(Involution):
    def __init__(self, xmax=1.0):
        Involution.__init__(self)
        self.xmax = xmax

    def __call__(self, x):
        return self.xmax - x

    def deriv(self, x):
        return -1.0


class FractionalInvolution(Involution):
    def __init__(self, xmax=1.0, k=2):
        Involution.__init__(self)
        self.xmax = xmax
        self.k = k

    def __call__(self, x):
        inv_k = 1.0/self.k
        return (self.xmax**inv_k - x**inv_k)**self.k

    def deriv(self, x):
        inv_k = 1.0/self.k
        return (self.xmax**inv_k - x**inv_k)**(self.k-1) * x**(inv_k - 1)
