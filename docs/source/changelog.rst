Change Log
==========

2023-10-29
----------
- Add qml_qaoa which supports running QML optimizers on expval function without any wrappers.
  This allows to run Quantum Natural Gradient (QNG) optimizer on QAOA circuits.

  .. code-block:: python

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qml_qaoa",
                "layers": 5,
                "optimizer": "qng",
                "optimizer_args": {
                    "stepsize": 0.0001,
                }
            },
            "params_inits": params_cofing
        },
        "problem": problem_config
    }
