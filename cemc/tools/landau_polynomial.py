import numpy as np
from scipy.optimize import minimize

class TwoPhaseLandauPolynomial(object):
    """Class for fitting a Landau polynomial to free energy data

    :param float c1: Center concentration for the first phase
    :param float c2: Center concentration for the second phase
    :param np.ndarray init_guess: Initial guess for the parameters
        The polynomial fitting is of the form
        A*(x - c1)^2 + B*(x-c2)*y^2 + C*y^4 + D*y^6
        This array should therefore contain initial guess
        for the four parameter A, B, C and D.
    """
    def __init__(self, c1=0.0, c2=1.0, num_dir=3, init_guess=None, conc_order=2):
        self.coeff = np.zeros(3)
        self.conc_coeff = np.zeros(conc_order+1)
        self.conc_order = conc_order
        self.c1 = c1
        self.c2 = c2
        self.init_guess = init_guess

    def equil_shape_order(self, conc):
        """Calculate the equillibrium shape concentration.
        
        :param float conc: Concentration
        """
        if abs(self.coeff[2] < 1E-8):
            n_eq = -0.5*self.coeff[0]*(conc - self.c2)/self.coeff[1]
            if n_eq < 0.0:
                return 0.0
            return n_eq
        delta = (self.coeff[1]/(3.0*self.coeff[2]))**2 - \
            self.coeff[0]*(conc-self.c2)/(3.0*self.coeff[2])

        if delta < 0.0:
            return 0.0
        
        n_eq = -self.coeff[1]/(3.0*self.coeff[2]) + np.sqrt(delta)
        if n_eq < 0.0:
            return 0.0
        return n_eq

    def grad_shape_coeff0(self, conc):
        """Return the gradient of the shape parameter with respect to the
           first coefficient."""
        n_eq = self.equil_shape_order(conc)
        if n_eq <= 0.0:
            return 0.0
        factor1 = -(conc-self.c2)/(3.0*self.coeff[2])
        factor2 = 1.0/np.sqrt((self.coeff[1]/(3.0*self.coeff[2]))**2 -
                              self.coeff[0]*(conc - self.c2)/(3.0*self.coeff[2]))
        return 0.5*factor1*factor2

    def grad_shape_coeff1(self, conc):
        """Return the gradient of the shape parameter with respect to the
            second coefficient."""
        n_eq = self.equil_shape_order(conc)
        if n_eq <= 0.0:
            return 0.0

        factor1 = self.coeff[1]/(9.0*self.coeff[2]**2)
        factor2 = 1.0/np.sqrt((self.coeff[1]/(3.0*self.coeff[2]))**2 -
                              self.coeff[0]*(conc - self.c2)/(3.0*self.coeff[2]))
        return - 1.0/(3.0*self.coeff[2]) + factor1*factor2

    def grad_shape_coeff2(self, conc):
        """Return the gradient of the equillibrium shape parameter
            with respect to the last."""
        n_eq = self.equil_shape_order(conc)
        if n_eq <= 0.0:
            return 0.0
        
        factor1 = 0.5/np.sqrt((self.coeff[1]/(3.0*self.coeff[2]))**2 -
                              self.coeff[0]*(conc - self.c2)/(3.0*self.coeff[2]))

        factor2 = self.coeff[0]*(conc - self.c2)/(3.0*self.coeff[2]**2) - \
            2*self.coeff[1]**2/(9.0*self.coeff[2]**3)

        return self.coeff[1]/(3.0*self.coeff[2]**2) + factor1*factor2

    def eval_at_equil(self, conc):
        """Evaluate the free energy at equillibrium order.
        
        :param float conc: Concentration
        """
        n_eq_sq = self.equil_shape_order(conc)
        return np.polyval(self.conc_coeff, conc - self.c1) + \
            self.coeff[0]*(conc - self.c2)*n_eq_sq + \
            self.coeff[1]*n_eq_sq**2 + \
            self.coeff[2]*n_eq_sq**3

    def fit(self, conc1, F1, conc2, F2):
        """Fit the free energy functional.
        
        :param numpy.ndarray conc1: Concentrations in the first phase
        :param numpy.ndarray F1: Free energy in the first phase
        :param numpy.ndarray conc2. Concentrations in the second phase
        :param numpy.ndarray F2: Free energy in the second phase
        """
        conc = np.concatenate((conc1, conc2))
        free_energy = np.concatenate((F1, F2))
        # S1 = np.sum(F1*(conc1 - self.c1)**2)
        # S2 = np.sum((conc1 - self.c1)**4)
        # A = S1/S2
        self.conc_coeff = np.polyfit(conc1 - self.c1, F1, self.conc_order)

        remains = F2 - np.polyval(self.conc_coeff, conc2 - self.c1)
        
        S1 = np.sum(remains*(conc2 - self.c2))
        S2 = np.sum((conc2 - self.c2)**2)
        B = S1/S2

        S1 = np.sum((conc2 - self.c2))
        S2 = np.sum((conc2 - self.c2)**2)
        K = S1/S2
        C = -B/(2.0*K)

        if self.init_guess is not None:
            x0 = self.init_guess
        else:
            x0 = np.array([B, C, 15.0])

        def mse(x):
            self.coeff = x
            pred = [self.eval_at_equil(conc[i]) for i in range(len(conc))]
            pred = np.array(pred)
            mse = np.mean((pred - free_energy)**2)
            return mse

        def jac(x):
            self.coeff = x
            n_eq_sq = np.array([self.equil_shape_order(conc[i])
                                for i in range(len(conc))])
            pred = [self.eval_at_equil(conc[i]) for i in range(len(conc))]
            pred = np.array(pred)
            grad = np.zeros(3)

            grad[0] = -2.0*np.mean((free_energy - pred)*((conc - self.c2)*n_eq_sq + grad_n_B))
            grad[1] = -2.0*np.mean((free_energy - pred)*n_eq_sq**2)
            grad[2] = -2.0*np.mean((free_energy - pred)*n_eq_sq**3)
            return grad

        res = minimize(mse, x0=x0, method="bfgs", jac=None)
        self.coeff = res["x"]
