Gate-based
==========

Gate-based solvers run parametrized quantum circuits on a universal (gate-model) quantum computer or simulator. QHyper implements the Quantum Approximate Optimization Algorithm (QAOA) and its variants.

The solvers are grouped by the platform they are implemented with:

* :doc:`PennyLane <solvers_gate_based_pennylane>` -- ``QAOA``, ``QML_QAOA``, ``WF_QAOA`` and ``H_QAOA`` that run on PennyLane simulators or on a QPU.
* :doc:`Qiskit <solvers_gate_based_qiskit>` -- ``QAOA`` that runs on IQM quantum computers (``platform: iqm``) or on the local Qiskit Aer simulator (``platform: qiskit_aer``).

.. toctree::
   :maxdepth: 2

   solvers_gate_based_pennylane
   solvers_gate_based_qiskit
