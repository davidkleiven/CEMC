from cemc.mcmc import ReactionPathSampler, PseudoBinaryConcRange
from cemc.mcmc import PseudoBinaryConcInitializer
from cemc.mcmc import PseudoBinarySGC


class PseudoBinaryReactPath(ReactionPathSampler):
    def __init__(self, mc_obj=None, react_crd=[0.0, 1.0], n_windows=10,
                 n_bins=10, data_file="reaction_path.h5"):

        if not isinstance(mc_obj, PseudoBinarySGC):
            raise TypeError("mc_obj has to be of type PseudoBinarySGC")

        cnst = PseudoBinaryConcRange(mc_obj)
        init = PseudoBinaryConcInitializer(mc_obj)
        ReactionPathSampler.__init__(
            self, mc_obj=mc_obj, react_crd=react_crd, react_crd_init=init,
            react_crd_range_constraint=cnst, n_windows=n_windows,
            n_bins=n_bins, data_file=data_file)
