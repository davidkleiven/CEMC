C++ Cluster Expansion Calculator
=================================

.. DANGER::
  After the atoms object have been attached to this calculator it must not
  be altered, except via the calculator's methods. So if you for instance
  want to change the composition, use
  :py:meth:`cemc.CE.set_composition`, not change
  the elements manually.

.. autoclass:: cemc.ce_calculator.CE
  :members:

.. automodule:: cemc.ce_calculator
  :members: get_ce_calc
