from cemc.mcmc import BiasPotential

class Strain(BiasPotential):
    def __init__(self, mc_sampler=None):
        from cemc.mcmc import FixedNucleusMC
        if not isinstance(mc_sampler, FixedNucleusMC):
            raise TypeError("mc_sampler has to be of type FixedNuceus sampler!")
        self.mc = mc_sampler
        self._volume = 10.0

    def __call__(self, system_changes):
        return 0.0