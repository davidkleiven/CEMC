from cemc.mcmc import ReactionCrdInitializer
from cemc.mcmc import BiasPotential
import numpy as np
import h5py as h5
import time
import os

class AdaptiveBiasPotential(BiasPotential):
    """Bias potential intended to be used by AdaptiveBiasReactionPathSampler

    :param list lim: List of length 2 with upper and lower bound of the 
        reaction coordinate. Ex. [0.0, 1.0]
    :param int n_bins: Number of bins used in the histogram
    :param float mod_factor: Modification factor. Each time
        the samplers is a bin, this factor is added to the 
        potential.
    :param ReactionCrdInitializer reac_init: Initializer to 
        which can obtain the reaction coordinate of 
        an arbitrary structure
    :param float T: Temperature in Kelvin
    :param Montecarlo mc: Monte Carlo object which samples
        the configurational space
    """
    def __init__(self, lim=[0.0, 1.0], n_bins=100, mod_factor=0.1,
                 reac_init=None, T=400, mc=None):
        from ase.units import kB
        self.xmin = lim[0]
        self.xmax = lim[1]
        self.nbins = n_bins
        self.bias_array = np.zeros(self.nbins)
        self.mod_factor = mod_factor
        self.reac_init = reac_init
        self.beta = 1.0/(kB*T)
        self.dx = (self.xmax - self.xmin)/self.nbins
        self.mc = mc

    def get_bin(self, value):
        """Return the bin corresponding to value.
        
        :param float value: Reaction coordinate
        :return: Corresponding bin
        :rtype: int
        """
        return int((value - self.xmin)*self.nbins/(self.xmax - self.xmin))

    def update(self):
        """Update the bias potential."""
        x = self.reac_init.get(None)
        bin_indx = self.get_bin(x)
        if bin_indx < 0 or bin_indx >= self.nbins:
            return
        cur_val = self.get_bias_potential(x)
        self.bias_array[bin_indx] += self.mod_factor
        new_val = self.get_bias_potential(x)

        # We need to make MC aware of that the 
        # energy of the current state changes
        # despite the fact that no move is performed
        self.mc.current_energy += (new_val - cur_val)

    def get_bias_potential(self, value):
        """Return the value of the bias potential.
        
        :param float value: Reaction coordinate
        :return: Value of the bias potential
        :rtype: float
        """
        bin_indx = self.get_bin(value)
        if bin_indx == self.nbins - 1:
            # Linear interpolation
            betaG2 = self.bias_array[bin_indx]
            betaG1 = self.bias_array[bin_indx-1]
            x1 = self.xmin + (bin_indx - 1)*self.dx
            betaG = (betaG2 - betaG1)*(value - x1)/self.dx + betaG1
        elif bin_indx == 0:
            # Linear interpolation
            betaG2 = self.bias_array[1]
            betaG1 = self.bias_array[0]
            x1 = self.xmin
            betaG = (betaG2 - betaG1)*(value - x1)/self.dx + betaG1
        else:
            # Perform quadratic interpolation
            x0 = self.xmin + bin_indx*self.dx + self.dx/2.0
            x_pluss = x0 + self.dx
            x_minus = x0 - self.dx
            x = np.array([x_minus, x0, x_pluss])
            X = np.zeros((3, 3))
            X[:, 0] = 1.0
            X[:, 1] = x
            X[:, 2] = x**2
            y = np.array([self.bias_array[bin_indx-1],
                          self.bias_array[bin_indx],
                          self.bias_array[bin_indx+1]])
            coeff = np.linalg.solve(X, y)
            betaG = coeff[0] + coeff[1]*value + coeff[2]*value**2
        return betaG/self.beta

    def __call__(self, system_changes):
        """Return the bias potential after changes are applied.

        :param system_changes: Changes to be applied. 
            Example: [(10, "Al", "Mg"), (20, "Mg", "Al)]
        :type system_changes: List of tuples

        :return: Bias potential after changes have been applied
        :rtype: float
        """
        # We require initializers that can 
        # get apply the system changes
        value = self.reac_init.get(self.mc.atoms, system_changes)
        return self.get_bias_potential(value)
        


class AdaptiveBiasReactionPathSampler(object):
    """Sample the free energy along a path by adaptively tuning a bias potential.

    :param Montecarlo mc_obj: Monte Carlo sampler
    :param ReactionCrdInitializer reac_crd_init: Initializer that 
        can both set and get the reaction coordinate of an arbitrary
        configuration. Benefitial if this support fast evaluation
        of the reaction coordinate if the proposed system changes
        are supplied.
    :param int n_bins: Number of bins
    :param str data_file: HDF5 file for data backup
    :param list reac_crd: List of length 2 with the upper
        and the lower value of the reaction coordinate.
        If it is possible for the MC sampler to leave the 
        region of interest, the MC sampler should have a 
        constraint prohibiting this attached.
    :param float mod_factor: Modification factor used to 
        update the bias potential
    :param float convergence_factor: If the bin that has been
        is least frequently visited have been visited more than
        convergence_factr*<average visits> the algorithm
        will consider the visit histogram as flat.
    :param int save_interval: Interval between writing backup to file 
        in seconds
    :param int log_msg_interval: Interval in seconds between every
        time a status message is printed.
    """
    def __init__(self, mc_obj=None, react_crd_init=None, n_bins=100, 
                 data_file="adaptive_bias_path_sampler.h5",
                 react_crd=[0.0, 1.0], mod_factor=0.1, convergence_factor=0.8,
                 save_interval=600, log_msg_interval=30):

        self.bias = AdaptiveBiasPotential(lim=react_crd, n_bins=n_bins, 
                                        mod_factor=mod_factor, 
                                        reac_init=react_crd_init, T=mc_obj.T,
                                        mc=mc_obj)
        self.mc = mc_obj
        self.mc.add_bias(self.bias)
        self.visit_histogram = np.zeros(n_bins, dtype=int)
        self.convergence_factor = convergence_factor
        self.save_interval = save_interval
        self.last_save = time.time()
        self.output_every = log_msg_interval
        self.current_step = 0.0
        self.current_min_val = 0.0
        self.current_max_val = 0.0
        self.average_visits = 0.0
        self.current_mc_step = 0
        self.last_visited_bin = 0
        self.data_file = data_file
        self.load_bias()

    def parameter_summary(self):
        """Print a summary of the current parameters."""
        self.log("Temperature: {}".format(self.mc.T))
        self.log("Modification factor: {}".format(self.bias.mod_factor))
        self.log("Reaction coordinate: [{}, {})".format(self.bias.xmin, self.bias.xmax))
        self.log("Save every: {} min".format(self.save_interval/60))
        self.log("Log message every: {} sec".format(self.output_every))

    def load_bias(self):
        """Try to load the bias potential from file."""
        if not os.path.exists(self.data_file):
            return
        
        with h5.File(self.data_file, 'r') as hfile:
            if "bias" in hfile.keys():
                data = np.array(hfile["bias"])
                self.bias.bias_array = data
        self.log("Bias loaded from {}".format(self.data_file))

        # Subtract of the an overall constant
        self.bias.bias_array -= self.bias.bias_array[0]

    def update(self):
        """Update the history."""
        self.bias.update()
        value = self.bias.reac_init.get(None)
        bin_indx = self.bias.get_bin(value)
        self.last_visited_bin = bin_indx
        self.visit_histogram[bin_indx] += 1

    def converged(self):
        mean = np.mean(self.visit_histogram)
        minval = np.min(self.visit_histogram)
        self.current_max_val = np.max(self.visit_histogram)
        self.current_min_val = minval
        self.average_visits = mean
        return minval > self.convergence_factor*mean

    def log(self, msg):
        """Log a message.

        :param str msg: Message to be logged
        """
        print(msg)

    def progress_message(self):
        """Output a progress message."""
        num_visited = np.count_nonzero(self.visit_histogram > 0)
        acc_rate = self.mc.num_accepted/self.mc.current_step
        self.log("Num MC steps: {}".format(self.current_mc_step))
        self.log("Visits: Min: {}. Max: {}. Average: {}. Num bins visited: {}. "
                 "Last visited bin: {}. Acc. ratio: {}"
                 "".format(self.current_min_val, self.current_max_val,
                           self.average_visits, num_visited, self.last_visited_bin,
                           acc_rate))

    def save(self):
        """Save records to the HDF5 file."""
        if os.path.exists(self.data_file):
            flag = "r+"
        else:
            flag = "w"

        with h5.File(self.data_file, flag) as hfile:
            if "visits" in hfile.keys():
                data = hfile["visits"]
                data[...] = self.visit_histogram
            else:
                hfile.create_dataset("visits", data=self.visit_histogram)

            if "bias" in hfile.keys():
                data = hfile["bias"]
                data[...] = self.bias.bias_array
            else:
                hfile.create_dataset("bias", data=self.bias.bias_array)

            if not "x" in hfile.keys():
                x = np.linspace(self.bias.xmin, self.bias.xmax, len(self.visit_histogram))
                hfile.create_dataset("x", data=x)
        self.log("Current state written to {}".format(self.data_file))

    def run(self):
        """Run simulation."""
        self.parameter_summary()
        conv = False
        now = time.time()
        while not conv:
            self.mc._mc_step()
            self.current_mc_step += 1
            self.update()
            if time.time() - now > self.output_every:
                self.progress_message()
                now = time.time()
            
            if time.time() - self.last_save > self.save_interval:
                self.save()
                self.last_save = time.time()
            conv = self.converged()



                