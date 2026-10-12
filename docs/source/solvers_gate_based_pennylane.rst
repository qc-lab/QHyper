PennyLane
=========

Available solvers:

* ``QAOA`` -- `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/abs/1411.4028>`_,
* ``WF_QAOA`` -- `Weight-free Quantum Approximate Optimization Algorithm <https://www.iccs-meeting.org/archive/iccs2023/papers/140770117.pdf>`_,
* ``H_QAOA`` -- hyper ``QAOA`` that also optimizes the penalty weights during training.

Available devices (``name`` is the name of a `PennyLane device <https://pennylane.ai/devices>`_):

* ``type: simulator`` -- a local simulator, e.g. ``default.qubit`` (recommended default), ``lightning.qubit``, ``default.mixed``, or the Qiskit simulators ``qiskit.aer`` and ``qiskit.basicsim``,
* ``type: qpu`` -- real quantum hardware available through a PennyLane plugin (e.g., ``qiskit.remote``) with the hardware ``backend`` name.

Example usage:

.. code-block:: yaml

    device:
        type: simulator
        name: default.qubit

.. note:: Devices other than the built-in ``default.*`` and ``lightning.qubit`` may require installing the corresponding PennyLane plugin, e.g. ``pennylane-qiskit`` for the ``qiskit.*`` devices (see :py:class:`~QHyper.devices.pennylane.PennyLaneDevice`).

.. autosummary::
   :toctree: generated/

   QHyper.solvers.gate_based.pennylane.qaoa.QAOA -- QAOA solver.
   QHyper.solvers.gate_based.pennylane.qml_qaoa.QML_QAOA -- QML QAOA solver.
   QHyper.solvers.gate_based.pennylane.wf_qaoa.WF_QAOA -- Weight Free QAOA solver.
   QHyper.solvers.gate_based.pennylane.h_qaoa.H_QAOA -- Hyper QAOA solver.
