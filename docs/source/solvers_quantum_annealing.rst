Quantum Annealing
=================

Quantum annealing solvers do not build quantum circuits. Instead, the problem is encoded as the energy landscape of a quadratic model, and a quantum annealer evolves towards its low-energy states, which correspond to good solutions.

Available solvers:

* ``Advantage`` -- `D-Wave Advantage quantum annealer <https://docs.dwavequantum.com/en/latest/quantum_research/index_about.html>`_ (QPU),
* ``CQM`` -- `D-Wave Constrained Quadratic Model Hybrid Solver <https://docs.dwavequantum.com/en/latest/concepts/models.html#concept-models-cqm>`_,
* ``DQM`` -- `D-Wave Discrete Quadratic Model Hybrid Solver <https://docs.dwavequantum.com/en/latest/concepts/models.html#concept-models-dqm>`_.

All of the above solvers run on D-Wave machines.

Example usage:

.. code-block:: yaml

    device:
        type: qpu
        name: DWaveSampler

.. note:: All the above solvers require a D-Wave `token <https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html>`_. It can be passed as ``device.token`` or set as the ``DWAVE_API_TOKEN`` environment variable.

.. autosummary::
   :toctree: generated/

   QHyper.solvers.quantum_annealing.dwave.cqm.CQM -- CQM solver.
   QHyper.solvers.quantum_annealing.dwave.dqm.DQM -- DQM solver.
   QHyper.solvers.quantum_annealing.dwave.advantage.Advantage -- Advantage solver.
