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

1. **Import QHyper:** Begin by importing the required elements from the library.

.. code-block:: python

    from QHyper.problems.knapsack import KnapsackProblem
    from QHyper.solvers import Solver


2. **Create problem:** Use the library's classes and functions to define and set up your problem.

.. code-block:: python

    problem = KnapsackProblem(max_weight=2, items=[(1, 2), (1, 2), (1, 1)])


3. **Define initial parameters:** Each solver requires initial parameters to start the optimization process.

.. code-block:: python

    params_config = {
        'angles': [[0.5]*5, [1]*5],
        'hyper_args': [1, 2.5, 2.5],
    }


4. **Create solver:** Use the library's classes to create a solver.

.. code-block:: python

    solver_config = {
        'solver': {
            'type': 'vqa',
            'args': {
                'optimizer': {
                    'type': 'scipy',
                    'maxfun': 200,
                },
                'pqc': {
                    'type': 'wfqaoa',
                    'layers': 5,
                },
            }
        },
        'hyper_optimizer': {
            'type': 'random',
            'processes': 5,
            'number_of_samples': 100,
            'bounds': [(1, 10), (1, 10), (1, 10)]
        }
    }
    vqa = Solver.from_config(problem, config=solver_config)


5. **Execute solver:** Run your experiments using the solver on defined problem.

.. code-block:: python

    solver_results = vqa.solve(params_config)
    print("Solver results:")
    print(f"Probabilities: {solver_results.probabilities}")
    print(f"Best params: {solver_results.params}")

    # Solver results:
    # Probabilities: {'00000': 0.0732912838324004, '00001': 0.01812365507384847, ...}
    # Best params: {'angles': array([[4.77452593, 3.29033494, 0.85409721, 2.25547951, 5.960884  ],
    #                               [1.64590219, 0.48733654, 0.26765959, 0.03158379, 3.06768805]]),
    #               'hyper_args': array([1. , 2.5, 2.5])}


6. **Evaluate and show results:** By using the `QHyper` library, you can easily evaluate and show the results of your experiments.

.. code-block:: python

    from QHyper.util import (
        weighted_avg_evaluation, sort_solver_results, add_evaluation_to_results)

    print("Evaluation:")
    print(weighted_avg_evaluation(
        solver_results.probabilities, problem.get_score,
        penalty=0, limit_results=10, normalize=True
    ))
    print("Sort results:")
    sorted_results = sort_solver_results(
        solver_results.probabilities, limit_results=10)

    results_with_evaluation = add_evaluation_to_results(
        sorted_results, problem.get_score, penalty=penalty)

    for result, (probability, evaluation) in results_with_evaluation.items():
        print(f"Result: {result}, "
            f"Prob: {probability:.5}, "
            f"Evaluation: {evaluation}")

    # Evaluation:
    # -2.1832776777678093
    # Sort results:
    # Result: 01101, Prob: 0.15204, Evaluation: -3
    # Result: 10101, Prob: 0.15204, Evaluation: -3
    # Result: 11001, Prob: 0.14235, Evaluation: -4
    # Result: 00110, Prob: 0.12695, Evaluation: -1
    # Result: 10010, Prob: 0.11914, Evaluation: -2
    # Result: 01010, Prob: 0.11914, Evaluation: -2
    # Result: 00000, Prob: 0.0644, Evaluation: 0
    # Result: 11111, Prob: 0.039469, Evaluation: 2
    # Result: 11110, Prob: 0.028006, Evaluation: 2
    # Result: 00010, Prob: 0.011519, Evaluation: 2


**Conclusion**

Congratulations! You've just scratched the surface of what the `QHyper` library
can offer. By following this guide, you've learned how to install the library,
embrace quantum algorithm and set up your initial
experiments.

.. For more advanced usage and examples, explore the
.. `demos <https://github.com/qc-lab/QHyper/tree/main/demo>` available on github.

Happy experimenting with `QHyper`!
