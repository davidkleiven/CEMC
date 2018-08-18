class BiasPotential(object):
    """
    Potential that can be used to manipulate the states visited.
    If added to MC run, the value returned from this will be
    added to the energy.
    """
    def __call__(self, system_changes):
        raise NotImplementedError("Has to be implemented in child classes!")
