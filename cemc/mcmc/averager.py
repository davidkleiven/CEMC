
class Averager(object):
    def __init__(self, ref_value=1.0):
        self._ref_value = float(ref_value)
        self._n_samples = 0.0
        self._mean = 0.0

    def __iadd__(self, value):
        """
        Implement += operator
        """
        if isinstance(value, Averager):
            self._mean += value._mean*(value._ref_value/self._ref_value)
            self._n_samples += value._n_samples
            return self

        self._n_samples += 1.0
        self._mean += value/self._ref_value
        return self

    def __add__(self, other):
        ratio = (other._ref_value/self._ref_value)
        new_obj = Averager(ref_value=self._ref_value)
        new_obj._mean = self._mean + other._mean*ratio
        new_obj._n_samples = self._n_samples+other._n_samples
        return new_obj

    def __itruediv__(self, number):
        return self.__idiv__(number)

    def __idiv__(self, number):
        self._mean /= float(number)
        self._n_samples /= float(number)
        return self

    def clear(self):
        """
        Clear the accumulated entries, but keep the reference value
        """
        self._n_samples = 0
        self._mean = 0.0

    @property
    def mean(self):
        if self._n_samples == 0:
            return self._mean*self._ref_value
        return (self._mean/self._n_samples)*self._ref_value
