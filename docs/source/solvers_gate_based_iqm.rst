IQM
===

Available solvers:

* ``QAOA`` -- `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/abs/1411.4028>`_ implemented with Qiskit, running on IQM quantum computers or on a local simulator.

Available devices:

* IQM Resonance (``name: iqm.resonance``) -- cloud access to IQM quantum computers:

  * ``type: qpu`` -- a real QPU selected with ``backend``: ``garnet``, ``emerald`` or ``sirius``. Requires an `IQM Resonance <https://resonance.meetiqm.com/>`_ ``token`` or the ``IQM_TOKEN`` environment variable,
  * ``type: simulator`` -- a local Qiskit Aer simulator.

* VLQ (``name: vlq``) -- the IQM quantum computer hosted by IT4Innovations and accessed through the LEXIS platform:

  * ``type: qpu`` -- requires the LEXIS ``project`` and ``resource_name``. Currently there is no public access to this machine,
  * ``type: simulator`` -- a local simulator of the 24-qubit VLQ architecture.

Example usage:

.. code-block:: yaml

    device:
        type: qpu
        name: iqm.resonance
        backend: garnet
        token: <YOUR_IQM_RESONANCE_TOKEN>

See the :doc:`IQM tutorial <user_guide/demo/iqm_tutorial>` for a complete example, and :py:class:`~QHyper.devices.iqm.ResonanceDevice` and :py:class:`~QHyper.devices.iqm.VLQDevice` for all device options.

.. autosummary::
   :toctree: generated/

   QHyper.solvers.gate_based.iqm.qaoa.QAOA -- IQM QAOA solver.
