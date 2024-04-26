Usage Guide
===========

Welcome to the usage guide for the `QHyper` library! This guide will provide you
with practical instructions on using the `QHyper` library for conducting
experiments. Whether you're new to quantum computing or an experienced
researcher, this guide will help you navigate the features and capabilities of the library.

Installation
------------

To get started, you'll need to install the `QHyper` library using pip. Open your
terminal or command prompt and execute the following command:

.. code-block:: bash

    pip install qhyper

Key Concepts
------------

- **Solvers:** `QHyper` is designed to facilitate the implementation
  and experimentation with different types of solvers. It is easy to create you
  own custom solvers and use it with other components already available in the library

- **Problems:** solvers interface was created to be compatible with any type of
  problem. You can use any problem from the `QHyper` library or create your own
  custom problem and use it with any solver from the library.

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
    

Solver configuration tutorial
---------------

This tutorial assumes following sample optimization problem definition:

.. code-block:: python

    problem:
        type: knapsack
        max_weight: 2
        items: [[1, 2], [1, 2], [1, 1]]

Which defines `knapsack problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_ of filling a knapsack with three items, each characterized with ``[weight cost]`` list. 
The goal is to put chosen items in the knapsack to achieve maximal cost  with total weight not exceeding  ``max_weight``

1. Basic  solver definition requires providing its type. Currently supported types are:

 * ``vga`` for variational algorithms
 * ``cqm`` for `D-Wave CQM <https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html#leap-s-hybrid-solvers>`_
 * ``dqm`` for `D-Wave DQM <https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html#leap-s-hybrid-solvers>`_
 * ``advantage`` for `D-Wave Advantage <https://docs.dwavesys.com/docs/latest/c_gs_4.html>`_ (currently default advantage_system5.4. is supported)

Sample code for defining type advantage solver

.. code-block:: python

    solver:
        type: advantage

2.  Configuring initial QUBO penalties (Lagrangian multipliers) 

``advantage`` solver requires problem definition in the `QUBO <https://arxiv.org/pdf/1811.11538>`_ form. QHyper automatically creates the QUBO for 
knapsack problem (for details see 
`Software Aided Approach for Constrained Optimization Based on QAOA Modifications <https://link.springer.com/chapter/10.1007/978-3-031-36030-5_10>`_. )

This, however, requires setting  three penalties (Lagrangian multipliers) i.e. hyperparameters  
for the cost function and two constraints: ensuring that problem encoding is correct and that knapsack weight fullfils ```max_weight``` requirement . 

In the example below, the constraint penalties  are set as ``hyper_args``

.. code-block:: python

    solver:
        type: advantage
        params_inits:
            hyper_args: [1, 2.5, 2.5]

3. Adding hyperoptimiser

Since guessing correct penalties is often difficult, there is also option to define ``hyper_optimiser`` to search for appropriate settings.
In the example below, ``grid`` search hyperoptimizer is applied to find  proper penalties  of the  knapsack optimized function.
The penalties are searched within specified  ``bounds`` with ``steps`` defined in the configuration. 
  

.. code-block:: python

    solver:
        type: advantage
        hyper_optimizer:
            type: grid
            steps: [0.01, 0.01, 0.01]
            bounds: [[1, 10], [1, 10], [1, 10]]

4. ``vqa`` solver type is a  set containing solvers based on  gate-based variational algorithms. Currenly `QAOA <https://arxiv.org/abs/1411.4028>`_, `WF-QAOA and H-QAOA <https://link.springer.com/chapter/10.1007/978-3-031-36030-5_10>`_
are supported obtained by setting ``pqc type`` to  ``qaoa``, ``wfqaoa`` and ``hqaoa`` repectively. 

Typical example of QAOA configuration is shown below. The parametrized quantum circuit is configured for  5 ``layers``.  Default local 
`Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_ ``optimizer``
from `Pennylane <https://pennylane.ai/>`_ (``type: qml``) with default options is used. 

Initial variational parameters optimised by Adam method are set as ``angles``.   Penalty weights are initialized  as ``hyper_args``.

.. code-block:: python

    solver:
        type: vqa
        pqc:
            type: qaoa
            layers: 5
        optimizer:
            type: qml
        params_inits:
            angles: [[0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1]]
            hyper_args: [1, 2.5, 2.5]


5. It is possible to further customized ``pqc`` with additional keyword arguments (see QHyper API documentation). Below example of setting `Pennylane simulator
type <https://pennylane.ai/plugins/>`_ for ``qaoa``  using ``backend`` keyword

.. code-block:: python

    solver:
        type: vqa
        pqc:
            type: qaoa
            layers: 5
            backend: default.qubit
        optimizer:
            type: qml
        params_inits:
            angles: [[0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1]]
            hyper_args: [1, 2.5, 2.5]


6. Customising ``optimizer`` settings is also possible. Below, more detailed sample configuration is shown. Please note that adding all 
native function options is possible (e.g. ``stepsize`` in this example are native 
from `Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_   ) 
 
.. code-block:: python
 
    solver:
        type: vqa
        pqc:
            type: qaoa
            layers: 5
        optimizer:
            type: qml
            optimizer: adam
            steps: 200
            stepsize: 0.005
        params_inits:
            angles: [[0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1]]
            hyper_args: [1, 2.5, 2.5]

7. It is also possible to make use of both ``optimizer`` and ``hyper_optimizer`` functionality. The example below is similar to that in point 6. 
However, as in point 3, penalties  are searched by ``hiper_optimizer`` within specified  ``bounds``. In this example it is done  by Cross Entropy Search  method (configured as ``cem``).  ``processes``, ``samples_per_epoch`` and ``epochs`` are parameters specific for ``cem``. 

.. code-block:: python
 
        solver:
        type: vqa
        pqc:
            type: wfqaoa
            layers: 5
        optmizer:
            type: qml
            optmizer: adam
            steps: 200
            stepsize: 0.005
        hyper_optimizer:
            type: cem
            processes: 4
            samples_per_epoch: 1000
            epochs: 10
            bounds: [[1, 10], [1, 10], [1, 10]]
        params_inits:
            angles: [[0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1]]
            hyper_args: [1, 2.5, 2.5]

8. Variety of (hyper)optimizers. In QHyper both ``hyper_optimizer`` and ``optimizer`` can be set up using keyword arguments given below. **Please note that additional keyword arguments for each** ``optimizer`` **or** ``hyper_optimizer`` **configuration can be taken directly from native  function definition (refer to indicated  API documentation).**
   
    *  ``qml``  customizable gradient descent set of optimizers from Pennylane  (see below) 
    * ``scipy``: `Scipy gradient descent set of optimizers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_  
    * ``basinhopping``: `Scipy global Basinhopping optimizer <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_   
    * ``random``: random optimizer (see QHyper API doc)   
    * ``grid``:  grid search optimizer (see QHyper API doc) 
    * ``cem``: Cross Entropy Optimizer (see QHyper API doc) 
    * ``dummy``: dummy optimizer (see QHyper API doc) 

  Additionally, ``qml`` set of optimizers can be further specified  (e.g. ``adam`` configuration was shown in point 6 above) using following keyword arguments (for details see `Pennylane documentation <https://docs.pennylane.ai/en/stable/introduction/interfaces.html#numpy>`_ ):
   
    * ``adam``: qml.AdamOptimizer,
    * ``adagrad``: qml.AdagradOptimizer,
    * ``rmsprop``: qml.RMSPropOptimizer,
    * ``momentum``: qml.MomentumOptimizer,
    * ``nesterov_momentum``: qml.NesterovMomentumOptimizer,
    * ``sgd``: qml.GradientDescentOptimizer,
    * ``qng``: qml.QNGOptimizer,



Conclusion
----------

Congratulations! You've just scratched the surface of what the `QHyper` library
can offer. By following this guide, you've learned how to install the library,
embrace quantum algorithm and set up your initial
experiments.

For more advanced usage and examples, explore the
`demos <https://github.com/qc-lab/QHyper/tree/main/demo>` available on github.

Happy experimenting with `QHyper`!
