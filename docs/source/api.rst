API
==================================

Here is the list of all the modules in the QHyper package. 

.. automodule:: QHyper
   :members:

.. rubric:: Simple Modules

.. autosummary::
   :toctree: generated

   polynomial -- Module that contains the class for a polynomial implementation
   constraint -- Module that implements the constraints
   parser -- Module for parsing from and to sympy (in the future there might be more formats)
   converter -- Module that contains the converter class with methods to convert a problem to a different form required by the solvers
   util -- Module that contains utility functions

.. rubric:: Core Modules

.. autosummary::
   :toctree: generated

   problems
   optimizers
   solvers
