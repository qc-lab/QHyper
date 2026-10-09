Qiskit
======

Available solvers:

* ``QAOA`` -- Classic `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/abs/1411.4028>`_ implemented with Qiskit.

The solver runs on one of the following platforms, selected with ``solver.platform``:

* ``iqm`` -- IQM quantum computers,
* ``qiskit_aer`` -- the local `Qiskit Aer <https://qiskit.github.io/qiskit-aer/>`_ simulator.

Available devices for ``platform: iqm``:

* IQM Resonance (``name: iqm.resonance``) -- cloud access to IQM quantum computers:

  * ``type: qpu`` -- a real QPU selected with ``backend``: ``garnet``, ``emerald`` or ``sirius``. Requires an `IQM Resonance <https://resonance.meetiqm.com/>`_ ``token`` or the ``IQM_TOKEN`` environment variable,
  * ``type: simulator`` -- a local IQM fake backend selected with ``backend``: ``adonis``, ``apollo``, ``aphrodite``, ``deneb`` or ``garnet`` (default).

* VLQ (``name: vlq``) -- the IQM quantum computer hosted by IT4Innovations and accessed through the LEXIS platform:

  * ``type: qpu`` -- requires the LEXIS ``project`` and ``resource_name``. Currently there is no public access to this machine,
  * ``type: simulator`` -- a local simulator of the 24-qubit VLQ architecture.

Available devices for ``platform: qiskit_aer``:

* ``type: simulator`` -- the local Qiskit Aer simulator. For now it is the only accepted type.

Example usage:

.. code-block:: yaml

    device:
        type: qpu
        name: iqm.resonance
        backend: garnet
        token: <YOUR_IQM_RESONANCE_TOKEN>

See the :doc:`IQM tutorial <user_guide/demo/iqm_tutorial>` for a complete example and :py:class:`~QHyper.devices.iqm.ResonanceDevice`, :py:class:`~QHyper.devices.iqm.VLQDevice` and :py:class:`~QHyper.devices.qiskit_aer.AerDevice` for all device options.

.. autosummary::
   :toctree: generated/

   QHyper.solvers.gate_based.qiskit.qaoa.QAOA -- Qiskit QAOA solver.
