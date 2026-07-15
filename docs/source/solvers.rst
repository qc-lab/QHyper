QHyper.solvers
==============

.. automodule:: QHyper.solvers

.. rubric:: Interface

.. autosummary::
   :toctree: generated/

   Solver -- Base class for solvers.
   SolverResult -- Dataclass for storing results.
   HyperOptimizer -- Wrapper for optimizing solver hyperparameters.
   Solvers -- Registry used to look up solver classes.

.. rubric:: Solver categories

.. toctree::
   :maxdepth: 2

   solvers_classical
   solvers_quantum_annealing
   solvers_gate_based
