import numpy as np
import warnings
warnings.simplefilter("ignore",np.RankWarning)

class SequencePredicter(object):
    def __init__(self, maxorder=10):
        self.maxorder = maxorder

    def view( self, x, y, poly):
        """
        Visualize the result of the prediction
        """
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        xfit = np.linspace(x[0],x[-1],100)
        ax.plot( xfit, poly(xfit) )
        ax.plot( x, y, "o", mfc="none" )
        return fig

    def __call__(self, x, y, view=False ):
        """
        Finds the optimal polynomial that best predicts the last entry of the
        y based on input from the first
        """
        if ( len(y) == 0 ):
            return 0.0,0.0
        if ( len(y) == 1 ):
            return y[0],0.0
        elif ( len(y) == 2 ):
            y_pred = y[0]
            return y_pred, np.abs(y_pred-y[1])

        #if ( len(y) > self.maxorder+1 ):
        #    data_y = y[-self.maxorder-1:-1]
        #    data_x = x[-self.maxorder-1:-1]

        data_y = y[:-1]
        data_x = x[:-1]
        cv = []
        y_predict = []
        polys = []
        for order in range(1,len(data_y)):
            coeff = np.polyfit( data_x, data_y, order )
            poly = np.poly1d(coeff)
            polys.append( poly )
            y_pred = poly(x[-1])
            cv.append( np.abs(y_pred-y[-1]) )
            y_predict.append( y_pred )
        indx = np.argmin(cv)
        if ( view ):
            fig = self.view( x, y, polys[indx] )
            return y_predict, cv[indx], fig
        return y_predict[indx], cv[indx]
