Monte Carlo Constraints
========================

If certain moves are not desired, CEMC supports constraints.
A constraint is simply a class that receives a MC trial move and tells
if it is an allowed move. So in order to set a specific constraint, simply
derive a new class from :py:class:`cemc.mcmc.mc_constraints.MCConstraint` and
implement the call method!

.. autoclass:: cemc.mcmc.mc_constraints.MCConstraint
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_constraints.PairConstraint
  :members:
  :special-members: __call__

.. autoclass:: cemc.mcmc.mc_constraints.FixedElement
  :members:
  :special-members: __call__
