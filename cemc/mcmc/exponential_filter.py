import numpy as np
from scipy.optimize import minimize

class ExponentialFilter(object):
    def __init__( self, min_time=1, max_time=10, n_subfilters=2, dt=1 ):
        self.min_time = min_time
        self.max_time = max_time
        self.filter_values = np.zeros(n_subfilters)
        self.tau = np.linspace( min_time, max_time, n_subfilters )
        self.dt = dt
        self.weights = np.exp(-self.dt/self.tau )
        self.n_samples = 0

    def get_normalization_factors(self):
        """
        Returns the normalization factors
        """
        #return self.tau*(1.0-np.exp(-self.n_samples*self.dt/self.tau) )/self.dt
        return (1.0-np.exp(-self.n_samples*self.dt/self.tau) )/(1.0-np.exp(-self.dt/self.tau))

    def reset(self):
        """
        Clears the content
        """
        self.filter_values = np.zeros_like(self.filter_values)
        self.n_samples = 0

    def add( self, value ):
        """
        Adds a new value
        """
        self.filter_values = self.filter_values*self.weights + value
        self.n_samples += 1

    def get( self ):
        """
        Get the filtered values
        """
        normalization = self.get_normalization_factors()
        return self.filter_values/normalization

    def get_std( self, std_value ):
        """
        Compute the standard deviation of the filtered value
        User has to supply the standard deviation of the individual values
        """
        normalization = self.get_normalization_factors()
        return std_value/np.sqrt(normalization)

    def status_msg( self, std_value=None ):
        """
        Returns a status message as a string
        """
        msg = "======== EXPONENTIAL FILTER INFORMATION ========\n"
        msg += "Number of samples: {}\n".format(self.n_samples)
        values = self.get()
        std = np.zeros_like(values)
        if ( std_value is not None ):
            std = self.get_std(std_value)
        for i in range(len(self.filter_values)):
            if ( std_value is None ):
                msg += "Decay time: {}. Value: {}\n".format(self.tau[i],values[i])
            else:
                msg += "Decay time: {}. Value: {} +- {} ({}\%)\n".format(int(self.tau[i]),values[i],std[i],100*std[i]/values[i])
        msg += "===============================================\n"
        return msg

    def slope( self, std_value=None ):
        """
        Estimate the filter slope
        """
        x = -self.tau/2.0
        y = self.get()
        var = 1/self.tau
        Sx = np.sum(x/var)
        Sy = np.sum(y/var)
        Sxx = np.sum(x**2 /var)
        Sxy = np.sum(x*y/var)
        Ss = np.sum(1.0/var)
        N = len(y)
        slope = (Ss*Sxy - Sx*Sy)/(Ss*Sxx - Sx**2)
        if ( std_value is None ):
            return slope
        std = self.get_std(std_value)
        var_slope = np.sum(std)/(N*Sxx - Sxy)
        return slope, np.sqrt(var_slope)

    def exponential_extrapolation( self ):
        """
        Returns the exponential extrapolated value at infinity
        This can be useful in ground state searches as one knows
        at low temperature the energy always decreases and the
        this value can be a measure of the true ground state energy
        """
        x = -self.tau/2.0
        y = self.get()
        x += x[-1]
        def cost( params ):
            constant = params[0]
            prefactor = params[1]
            damping = params[2]
            predict = constant + prefactor*np.exp(-damping*x)
            return np.sum( (y-predict)**2 )
        res = minimize( cost, x0=(y[0],0.0,0.0) )
        params = res["x"]
        return params[0]
