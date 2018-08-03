Monte Carlo Observers
=======================

Monte Carlo observers are callable classes that can be attached to any
Monte Carlo sampler via :py:meth:`cemc.mcmc.Montecarlo.attach`.
The *__call__* method of the attached classes gets executed
on even intervals specied by the *interval* argument in the *attach* function.
To create your own observers, you can simply derive a new class
from the :py:class:`cemc.mcmc.mc_observers.MCObserver` and attach it to
the Monte Carlo sampler!

.. autoclass:: cemc.mcmc.mc_observers.MCObserver
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.CorrelationFunctionTracker
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.PairCorrelationObserver
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.LowestEnergyStructure
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.SGCObserver
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.Snapshot
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.NetworkObserver
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.NetworkObserver
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_observers.SiteOrderParameter
  :members:
  :special-members: __call__
