import numpy as np

class SequencePredicter(object):
    def __init__(self, maxorder=10):
        self.maxorder = maxorder

    def __call__(self, x, y ):
        """
        Finds the optimal polynomial that best predicts the last entry of the
        y based on input from the first
        """
        if ( len(y) < 2 ):
            raise ValueError( "The length of the provided array has to be larger than 2!" )

        if ( len(y) == 2 ):
            y_pred = y[0]
            return y_pred, np.abs(y_pred-y[1])

        if ( len(y) > self.maxorder+1 ):
            data_y = y[-self.maxorder-1:-1]
            data_x = x[-self.maxorder-1:-1]
        else:
            data_y = y[:-1]
            data_x = x[:-1]
        cv = []
        y_predict = []
        print (data_x,data_y)
        for order in range(1,len(data_y)):
            coeff = np.polyfit( data_x, data_y, order )
            poly = np.poly1d(coeff)
            y_pred = poly(x[-1])
            cv.append( np.abs(y_pred-y[-1]) )
            y_predict.append( y_pred )
        indx = np.argmin(cv)
        return y_predict[indx], cv[indx]
