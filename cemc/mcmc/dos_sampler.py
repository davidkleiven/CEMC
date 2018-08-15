from cemc.mcmc import CompositionDOS
import numpy as np
import time


class SGCCompositionFreeEnergy(object):
    """Class holds the nessecary information required to sample DOS."""
    def __init__(self, nbins=100, hist_limits=[]):
        self.nbins = nbins
        self.dim_names = [hist_lim[0] for hist_lim in hist_limits]
        self.hist_limits = [(hist_lim[1], hist_lim[2]) for hist_lim
                            in hist_limits]
        if len(hist_limits) == 1:
            self.histogram = np.zeros(self.nbins, dtype=int)
        elif len(hist_limits) == 2:
            self.histogram = np.zeros((self.nbins, self.nbins), dtype=int)
        else:
            raise ValueError("Only 1D and 2D histograms are supported!")

    def reset(self):
        """Reset the sampler."""
        if len(self.hist_limits) == 1:
            self.histogram[:] = 0
        elif len(self.hist_limits) == 2:
            self.histogram[:, :] = 0

    def get_max_std(self):
        """
        Returns the estimated max variance among the
        sites that has been visited
        """
        flattened = self.histogram.flatten()
        flattened = flattened[flattened > 0]
        var = 1.0 / np.sqrt(flattened)
        if len(var) == 0:
            return 1.0
        elif len(var) <= 2:
            return np.min(var)
        perc = np.percentile(var, 1)
        return perc

    def get_one_index(self, value, dim=0):
        """Get index along one direction."""
        indx = (value - self.hist_limits[dim][0]) * self.nbins
        indx /= (self.hist_limits[dim][1] - self.hist_limits[dim][0])
        return int(indx)

    def get_indx(self, singlets):
        """
        Compute the index to the multidimensional histogram
        """
        indices = [self.get_one_index(singlets[dim], dim=dim) for dim in
                   range(len(self.hist_limits))]
        return indices

    def valid_index(self, indices):
        """Check if the indices is valid."""
        indices = np.array(indices)
        if np.any(indices < 0):
            return False
        if np.any(indices >= self.nbins):
            return False
        return True

    def update_histogram(self, singlets):
        """Update the histogram"""
        indices = self.get_indx(singlets)
        if not self.valid_index(indices):
            return

        # Convert to column vector
        indices = np.reshape(indices, (len(indices), 1))
        self.histogram[indices] += 1

    def find_dos(self, sgc_sampler, max_rel_unc=0.01, min_num_steps=0,
                 rettype="comp_dos_obj"):
        """Run Monte Carlo."""
        sgc_sampler.estimate_correlation_time()
        sgc_sampler.equillibriate()
        maxiter = 100000000
        counter = 0
        check_conv_every = 1000
        output_every = 30
        max_std = -1
        last_time = time.time()
        supported_ret_types = ["comp_dos_obj", "raw_hist"]
        if rettype not in supported_ret_types:
            msg = "Return type has to be one of {}".format(supported_ret_types)
            raise ValueError(msg)
        while counter < maxiter:
            counter += 1
            sgc_sampler.reset()
            if counter > 1:
                sgc_sampler.is_first = False
            sgc_sampler._mc_step()
            singlets = sgc_sampler.averager.singlets
            assert sgc_sampler.averager.counter == 1
            # singets_array = [singlets[key] for key in self.dim_names]
            self.update_histogram(singlets)
            if counter % check_conv_every == 0:
                max_std = self.get_max_std()
                if max_std < max_rel_unc and counter > min_num_steps:
                    msg = "Composition Free Energy converged!."
                    msg += "Max std: {}".format(max_std)
                    print(msg)
                    if rettype == "comp_dos_obj":
                        return CompositionDOS(self.hist_limits, self.histogram)
                    else:
                        return self.hist_limits, self.histogram
            if time.time() - last_time > output_every:
                msg = "Current number of steps: {}. ".format(counter)
                msg += "Max std: {}.".format(max_std)
                print(msg)
                last_time = time.time()
