Changelog
=========

2023-11-13
----------
- QAOA can now create cost operator from HOBO

2023-11-11
----------
- Add params to the history of optimization, also all the results are stored in the history

2023-11-08
----------
- Allow to pass any keyword arguments to SciPy optimizer

2023-11-06
----------
- Add history to SolverResult and OptimizerResult

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
            "params_inits": params_config
        },
        "problem": problem_config
    }
