Classical
=========

Classical solvers do not use quantum hardware, the problem is solved with classical optimization methods running on a CPU instead of a quantum processing unit (QPU). 

Available solvers:

* ``Gurobi`` -- classical `Gurobi Optimizer <https://www.gurobi.com/solutions/gurobi-optimizer/>`_.

Example usage:

.. code-block:: yaml

    solver:
        category: classical
        platform: gurobi
        name: Gurobi

.. note:: For larger problem instances a Gurobi `license <https://www.gurobi.com/solutions/licensing/>`_ is required.

.. autosummary::
   :toctree: generated/

   QHyper.solvers.classical.gurobi.Gurobi -- Gurobi solver.
