import numpy as np
class HistogramConvergenceCheck( object ):
    def __init__( self ):
        pass

    def is_converged( self, histogram ):
        """
        Returns True if the histogram has converged
        """
        raise NotImplementedError( "is_converged has to be implemented in derived classes" )

class FlatHistgoram( HistogramConvergenceCheck ):
    """
    The histogram is considered to be converged if
    min(histogram) > fraction_of_mean*mean(histogram)
    bins with 0 entries are ignored
    """
    def __init__( self, fraction_of_mean=0.95 ):
        HistogramConvergenceCheck(self)
        self.fraction_of_mean = fraction_of_mean

    def is_converged( self, histogram ):
        """
        Returns True if the histogram is converged
        """
        masked_hist = histogram[histogram>0]
        mean = np.mean(masked_hist)
        return np.all( masked_hist > self.fraction_of_mean*mean )

class GrowthFluctuation( HistogramConvergenceCheck ):
    """
    The histogram is considered converged if the value in each bin
    is larger than a factor times the fluctuation in that bin
    """
    def __init__( self, factor=5000 ):
        HistogramConvergenceCheck.__init__(self)
        self.factor = factor

    def is_converged( self, histogram, fluctuations ):
        raise NotImplementedError( "Not implemented yet!" )
