Installation & First Steps
==========================

Installation
------------

To get started, you'll need to install the `QHyper` library using pip. Open your
terminal or command prompt and execute the following command:

.. code-block:: bash

    pip install qhyper

.. Key Concepts
.. ------------
..
.. - **Solvers:** `QHyper` is designed to facilitate the implementation
..   and experimentation with different types of solvers. It is easy to create you
..   own custom solvers and use it with other components already available in the library
..
.. - **Problems:** solvers interface was created to be compatible with any type of
..   problem. You can use any problem from the `QHyper` library or create your own
..   custom problem and use it with any solver from the library.

Getting Started
---------------

Once you've installed the library, you're ready to dive into experimenting using `QHyper`.
Here are some key steps to get started:

1. **Define solver config and initial parameters:**
Check :doc:`../api` for available solvers and problems. You can define your own
solvers, problems and optimizer - for that checkout :doc:`demo/different_configurations`.

.. code-block:: python

    params_config = {
        'angles': [[0.5]*5, [1]*5],
        'hyper_args': [1, 2.5, 2.5],
    }

    solver_config = {
        'optimizer': {
            'type': 'scipy',
            'maxfun': 200,
            "bounds": [(0, 2 * 3.1415)] * 10
        },
        'pqc': {
            'type': 'wfqaoa',
            'layers': 5,
        },
        'hyper_optimizer': {
            'type': 'random',
            'processes': 4,
            'number_of_samples': 100,
            'bounds': [(1, 10), (1, 10), (1, 10)]
        },
        'params_inits': params_config
    }


2. **Create solver:** Use the library's classes to create a solver.

.. code-block:: python

    from QHyper.solvers import VQA
    from QHyper.problems.knapsack import KnapsackProblem

    problem = KnapsackProblem(max_weight=2, items_weights=[1, 1, 1], items_values=[2, 2, 1])

    vqa = VQA.from_config(problem, config=solver_config)


3. **Create solver from config:**
You can also create a solver from a config file. That's also suggested way to create a solver.

.. code-block:: python

    from QHyper.solvers import solver_from_config

    problem_config = {
        'type': 'knapsack',
        'max_weight': 2,
        'items_weights': [1, 1, 1],
        'items_values': [2, 2, 1],
    }

    full_solver_config = {
        "solver": {
            "type": "vqa",
            **solver_config
        },
        "problem": problem_config
    }
    vqa = solver_from_config(full_solver_config)


5. **Execute solver:** Run your experiments using the solver on defined problem.

.. code-block:: python

    solver_results = vqa.solve(params_config)
    print("Solver results:")
    print(f"Probabilities: {solver_results.probabilities}")
    print(f"Best params: {solver_results.params}")

    # Solver results:
    # Probabilities: [(0, 0, 0, 0, 0, 0.00392139), (0, 0, 0, 0, 1, 0.01346938),
    #                 (0, 0, 0, 1, 0, 0.05722635), (0, 0, 0, 1, 1, 0.0166838 ),
    #
    #                 (1, 1, 1, 1, 0, 0.02977723), (1, 1, 1, 1, 1, 0.02197872)]
    #
    # Best params: {'angles': array([0.27298414, 2.2926187 , 0.        , 0.76391714, 0.15569598,
    #                                0.4237506 , 0.93474157, 1.39996954, 1.38701602, 0.36818742]),
    #               'hyper_args': array([8.77582845, 7.32430447, 1.02777043])}


6. **Evaluate and show results:**
By using the `QHyper` library, you can easily evaluate and show the results of your experiments.

.. code-block:: python

    from QHyper.util import (
        weighted_avg_evaluation, sort_solver_results, add_evaluation_to_results)

    problem = vqa.problem

    print("Evaluation:")
    print(weighted_avg_evaluation(
        solver_results.probabilities, problem.get_score,
        penalty=0, limit_results=10, normalize=True
    ))
    print("Sort results:")
    sorted_results = sort_solver_results(
        solver_results.probabilities, limit_results=10)
    print(sorted_results)

    results_with_evaluation = add_evaluation_to_results(
        sorted_results, problem.get_score, penalty=0)

    for rec in results_with_evaluation:
        print(f"Result: {rec}, "
            f"Prob: {rec['probability']:.5}, "
            f"Evaluation: {rec['evaluation']:.5}")

    # Evaluation:
    # -1.669217721264391

    # Sort results:
    # Sorted results:
    # [(1, 1, 0, 0, 1, 0.14605589) (1, 0, 1, 0, 1, 0.09231208)
    #  (0, 1, 1, 0, 1, 0.09231208) (1, 0, 1, 1, 0, 0.06831021)
    #  (0, 1, 1, 1, 0, 0.06831021)]

    # Result: (1, 1, 0, 0, 1, 0.14605589, -4.), Prob: 0.14606, Evaluation: -4.0
    # Result: (1, 0, 1, 0, 1, 0.09231208, -3.), Prob: 0.092312, Evaluation: -3.0
    # Result: (0, 1, 1, 0, 1, 0.09231208, -3.), Prob: 0.092312, Evaluation: -3.0
    # Result: (1, 0, 1, 1, 0, 0.06831021, 0.), Prob: 0.06831, Evaluation: 0.0
    # Result: (0, 1, 1, 1, 0, 0.06831021, 0.), Prob: 0.06831, Evaluation: 0.0


**Conclusion**

Congratulations! You've just scratched the surface of what the `QHyper` library
can offer. By following this guide, you've learned how to install the library,
embrace quantum algorithm and set up your initial
experiments.

Happy experimenting with `QHyper`!
