
class GradCoeffEvaluator(object):
    def __init__(self):
        pass

    def evaluate(self, x, free_var):
        raise NotImplementedError("Has to be implemented in derived classes!")

    def derivative(self, x, free_var):
        return np.ones(len(x))
