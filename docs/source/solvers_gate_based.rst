Gate-based
==========

Gate-based solvers run parametrized quantum circuits on a universal (gate-model) quantum computer or simulator. QHyper implements the Quantum Approximate Optimization Algorithm (QAOA) and its variants.

The solvers are grouped by the platform they are implemented with:

* :doc:`PennyLane <solvers_gate_based_pennylane>` -- ``QAOA``, ``QML_QAOA``, ``WF_QAOA`` and ``H_QAOA`` that run on PennyLane simulators or on a QPU.
* :doc:`IQM <solvers_gate_based_iqm>` (``platform: iqm``) -- ``QAOA`` that runs on a local simulator or on IQM quantum computers (IQM Resonance, VLQ).

.. toctree::
   :maxdepth: 2

   solvers_gate_based_pennylane
   solvers_gate_based_iqm
