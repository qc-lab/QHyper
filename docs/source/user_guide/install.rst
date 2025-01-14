Installation & first steps
==========================

Installation
------------

To get started, install the `QHyper` library using pip. Open your terminal or command prompt and run the following command (make sure you are using Python 3.12):

.. code-block:: bash

    pip install qhyper


Getting Started
---------------

1. **Import an optimization problem:**

Here, we will solve the Knapsack Problem, but check out the :doc:`API<../generated/QHyper.problems>` for other available problems.

.. code-block:: python

    from QHyper.problems.knapsack import KnapsackProblem
    problem = KnapsackProblem(max_weight=2,
                              item_weights=[1, 1, 1],
                              item_values=[2, 2, 1])


2. **Create a solver:** 

Use the library's classes to create a solver.

.. code-block:: python

    from QHyper.solvers.gate_based.pennylane import QAOA
    from QHyper.optimizers import OptimizationParameter
    from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent

    solver = QAOA(problem,
        layers=5,
        gamma=OptimizationParameter(init=[0.25, 0.25, 0.25, 0.25, 0.25]),
        beta=OptimizationParameter(init=[-0.5, -0.5, -0.5, -0.5, -0.5]),
        optimizer=QmlGradientDescent(),
        penalty_weights=[1, 2.5, 2.5],
    )


3. **Run experiments:** 

Run the solver.

.. code-block:: python

    solver_results = solver.solve()

4. **Show the results:** 

Sort and display top 5 results.

.. code-block:: python

    from QHyper.util import sort_solver_results

    sorted_results = sort_solver_results(
        solver_results.probabilities, limit_results=5)

    print(sorted_results.dtype.names)
    for result in sorted_results:
        print(result)

    # ('x0', 'x1', 'x2', 'x3', 'x4', 'probability')
    # (1, 1, 0, 0, 1, 0.24827694)
    # (0, 1, 1, 0, 1, 0.18271937)
    # (1, 0, 1, 0, 1, 0.18271937)
    # (1, 1, 1, 0, 1, 0.15528488)
    # (1, 1, 1, 1, 1, 0.03339847)


**Summary**

You have successfully installed the QHyper library and set up your first experiment. 

Check out more advanced the tutorials: :doc:`solver_configuration`, :doc:`demo/typical_use_cases`, and :doc:`demo/defining_problems`.
