import time


class MultithreadPerformance(object):
    """
    Class for testing performance using multithreading for MC
    updates.

    :param int max_threads: Maximum number of threads to use
    """
    def __init__(self, max_threads=1):
        self.max_threads = max_threads

    def run(self, mc=None, num_mc_steps=10000):
        """
        Run the test starting from one thread.

        Caveat 1: The work done is different depending of a MC
        move is accepted or rejected. To only test accepted moves,
        it is recommended to set a unrealistically high temperature
        in the mc object (for instace 10000000K).

        Caveat 2: To ensure that the major workload is actually associated
        with MC moves, it is important that the number of MC steps is
        sufficiently high.

        :param cemc.mcmc.Montecarlo mc: MC instance
        :param int num_mc_steps: Number of MC steps to use
        """
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
            time_per_step = 1000*(end - start)/num_mc_steps
            execution_times.append(time_per_step)

        print()
        for i, exec_time in enumerate(execution_times):
            print("Num. treads: {}. Exec time per MC step: {}".format(i+1, exec_time))
        

