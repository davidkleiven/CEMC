import time


class MultithreadPerformance(object):
    """
    Class for testing performance using multithreading for MC
    updates.
    """
    def __init__(self, max_threads=1):
        self.max_threads = max_threads

    def run(self, mc=None, num_mc_steps=10000):
        from cemc.mcmc import Montecarlo
        if not isinstance(mc, Montecarlo):
            raise TypeError("mc has to be of type Montecarlo")

        execution_times = []
        for num_threads in range(1, self.max_threads+1):
            print("Using {} threads for CF update".format(num_threads))
            mc.atoms.get_calculator().set_num_threads(num_threads)
            start = time.time()
            mc.runMC(steps=num_mc_steps, equil=False, mode='fixed')
            end = time.time()
            time_per_step = (end - start)/num_mc_steps
            execution_times.append()

            print("Time per mc step: {} ms".format(time_per_step*1000))
