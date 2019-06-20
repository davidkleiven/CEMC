Multithreading
===========================================================

CEMC can internally use multithreading to accelerate the update of 
correlation functons. The number of threads can be set

>>> atoms.get_calculator().set_num_threads(4)

if the CE calculator has been attached to the Atoms object. 
However, multithreading is not effective in all cases. To check 
the performance as a function of the number of threads for a given 
Monte Carlo calculations, the MultithreadingPerformance class can be used.
The efficiency of multithreading acceleration mainly depend on the number of
ECIs and the cluster sizes.

>>> from cemc.tools import MultithreadPerformance
>>> performance_monitor = MultithreadPerformance(max_threads=8)
>>> performance.run(mc=mc, num_mc_steps=100000)

where **mc** is an **Montecarlo** instance. The output for an example case 
run on *Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz*

| Num. treads: 1. Exec time per MC step: 0.5708758211135865 ms
| Num. treads: 2. Exec time per MC step: 0.3189826035499573 ms
| Num. treads: 3. Exec time per MC step: 0.25949653387069704 ms
| Num. treads: 4. Exec time per MC step: 0.2173831868171692 ms
| Num. treads: 5. Exec time per MC step: 0.20085367918014527 ms
| Num. treads: 6. Exec time per MC step: 0.2042735981941223 ms
| Num. treads: 7. Exec time per MC step: 0.19945097208023072 ms
| Num. treads: 8. Exec time per MC step: 0.21075201749801636 ms

In this case 4 threads seems to be a good choice.

.. autoclass:: cemc.tools.MultithreadPerformance
   :members: