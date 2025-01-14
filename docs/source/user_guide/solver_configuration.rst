=============================
Solver configuration tutorial
=============================

Solver types
------------

| The basic solver definition requires the specification of its type. 
| Currently supported solver types are:

* quantum annealing

    * :py:class:`Advantage<.Advantage>` for the `D-Wave Advantage Solver <https://docs.dwavesys.com/docs/latest/c_gs_4.html>`_ (currently the default advantage_system5.4. is supported);
    * :py:class:`CQM<.CQM>` for the `D-Wave Constrained Quadratic Model Hybrid Solver <https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html#cqm-sdk>`_;
    * :py:class:`DQM<.DQM>` for the `D-Wave Discrete Quadratic Model Hybrid Solver <https://docs.ocean.dwavesys.com/en/stable/concepts/dqm.html#dqm-sdk>`_;
    * `Note`: for all the above solvers the D-Wave `token <https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html>`_ is required.

* gate-based

    * :py:class:`QAOA<.QAOA>` for the `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/abs/1411.4028>`_;
    * :py:class:`WF_QAOA<.WF_QAOA>` for the `Weight-free Quantum Approximate Optimization Algorithm <https://www.iccs-meeting.org/archive/iccs2023/papers/140770117.pdf>`_;

* classical

    * :py:class:`Gurobi<.Gurobi>` for the classical `Gurobi Optimizer <https://www.gurobi.com/solutions/gurobi-optimizer/>`_;
    * `Note`: for larger problem instances Gurobi `license <https://www.gurobi.com/solutions/licensing/>`_ is required.

Problem definition
------------------

This tutorial assumes the following sample optimization problem definition:

.. tabs::

    .. code-tab:: python

        from QHyper.problems.knapsack import KnapsackProblem
        problem = KnapsackProblem(max_weight=2, 
                                  item_weights=[1, 1, 1],
                                  item_values=[2, 2, 1])

    .. code-tab:: yaml

        problem:
            type: KnapsackProblem
            max_weight: 2
            item_weights: [1, 1, 1]
            item_values: [2, 2, 1]

    .. code-tab:: py JSON

        {
            "problem": { 
                    "type": "KnapsackProblem",
                    "max_weight": 2,
                    "item_weights": [1, 1, 1],
                    "item_values": [2, 2, 1],
                }
        }


This specifies the `Knapsack Problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_: fill a knapsack with three items, each characterized with a weight and cost, to maximize the total value without exceeding ``max_weight``.


Configuring quantum annealing solvers: D-Wave
---------------------------------------------


The initial penalty weights for constrained problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some solvers, such as the `D-Wave Advantage` hybrid solver, require the problem definition in the `Quadratic Unconstrained Binary Optimization (QUBO) <https://arxiv.org/abs/1811.11538>`_ form. QHyper automatically creates the QUBO, e.g., for the Knapsack Problem:

.. math::
   f(\boldsymbol{x}, \boldsymbol{y}) =
   - \alpha_0 \underbrace{\sum_{i = 1}^N c_i x_i}_{\text{cost function}} + \alpha_1 \underbrace{(1 - \sum_{i=1}^W y_i)^2}_{\text{constraint encoding}} + \alpha_2 \underbrace{(\sum_{i=1}^W iy_i - \sum_{i=1}^N w_ix_i)^2}_{\text{weight constraint}},

where
 * :math:`\alpha_j` are the penalty weights  (i.e. Lagrangian multipliers, hyperparameters of the optimized function);
 * :math:`N=3` is the number of available items;
 * :math:`W=` ``max_weight`` is the maximum weight of the knapsack;
 * :math:`c_i` and :math:`w_i` are the values and weights specified in ``item_values`` and ``item_weights`` lists of the configuration;
 * The goal is to optimize :math:`\boldsymbol{x} = [x_i]_N` which is a Boolean vector, where :math:`x_i = 1`  if and only if the item :math:`i` was selected to be inserted into the knapsack;
 * :math:`\boldsymbol{y} = [y_i]_W` is a one-hot vector where :math:`y_i = 1` if and only if the weight of the knapsack is equal to :math:`i`.


To define the function properly, you need to set three penalty terms :math:`\alpha_j`, which act as hyperparameters.
These penalties are used to combine the cost function and constraints. The first constraint ensure that the problem encoding is correct, and the second  that the total weight in the knapsack does not exceed the ``max_weight`` limit.


D-Wave Advantage solver
^^^^^^^^^^^^^^^^^^^^^^^

In the example below, the solver used is the D-Wave Advantage quantum annealing system and the constraint penalties (:math:`\alpha_j`) are set using the ``penalty_weights`` keyword argument. The ``num_reads`` argument is the amount of samples.

.. tabs::

    .. code-tab:: python

        from QHyper.solvers.quantum_annealing.dwave import Advantage

        solver = Advantage(problem, 
                           penalty_weights=[1, 2.5, 2.5],
                           num_reads=10)

    .. code-tab:: yaml

        solver:
            category: quantum_annealing
            platform: dwave
            name: Advantage
            penalty_weights: [1, 2.5, 2.5]
            num_reads: 10

    .. code-tab:: json

        {
            "solver": {
                "category": "quantum_annealing",
                "platform": "dwave",
                "name": "Advantage",
                "penalty_weights": [1, 2.5, 2.5],
                "num_reads": 10
            }
        }


Adding a hyperoptimizer
^^^^^^^^^^^^^^^^^^^^^^^

| Since guessing the correct penalty weights is often a difficult task, there is also an option to define a :py:class:`HyperOptimizer<.HyperOptimizer>` to search for the appropriate settings.

| In the example below, :py:class:`GridSearch<.GridSearch>` optimizer is applied to find the proper penalty weights for the knapsack QUBO formulation. The penalty weights are searched within specified  bounds (``min``, ``max``)  and incremented by a specified ``step`` size.

.. tabs::

    .. code-tab:: python

        from QHyper.solvers.hyper_optimizer import HyperOptimizer
        from QHyper.optimizers.grid_search import GridSearch
        from QHyper.solvers.quantum_annealing.dwave import Advantage

        hyper_optimizer = HyperOptimizer(
            optimizer=GridSearch(), 
            solver=Advantage(problem),
            penalty_weights={"min": [1, 1, 1], "max": [2.1, 2.1, 2.1], "step": [1, 1, 1]}
        )


    .. code-tab:: yaml

        solver:
            category: quantum_annealing
            platform: dwave
            name: Advantage
        hyper_optimizer:
            optimizer: 
                type: GridSearch
            penalty_weights: 
                min: [1, 1, 1]
                max: [2.1, 2.1, 2.1]
                step: [1, 1, 1]

    .. code-tab:: json

        {
            "solver": {
                "category": "quantum_annealing",
                "platform": "dwave",
                "name": "Advantage"
            },
            "hyper_optimizer": {
                "optimizer": {
                    "type": "GridSearch"
                },
                "penalty_weights": {
                    "min": [1, 1, 1],
                    "max": [2.1, 2.1, 2.1],
                    "step": [1, 1, 1]
                }
            }
        }



Configuring gate-based solvers: QAOA
------------------------------------

| A typical example of the QAOA configuration is presented below. 
| The quantum circuit consists of 5 ``layers``. The variational parameters ``gamma`` and ``beta`` are specified using ``OptimizationParameters``.
| A local :py:class:`QmlGradientDescent<.QmlGradientDescent>` ``optimizer`` (by default `Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_) with the default settings is used.
| Problem's penalty weights are defined in ``penalty_weights``.

.. tabs::

    .. code-tab:: python

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

    .. code-tab:: yaml

        solver:
            category: gate_based
            platform: pennylane
            name: QAOA
            layers: 5
            gamma:
                init: [0.25, 0.25, 0.25, 0.25, 0.25]
            beta:
                init: [-0.5, -0.5, -0.5, -0.5, -0.5]
            optimizer: 
                type: QmlGradientDescent
            penalty_weights: [1, 2.5, 2.5]

    .. code-tab:: json
        
        {
            "solver": {
                "category": "gate_based",
                "platform": "pennylane",
                "name": "QAOA",
                "layers": 5,
                "gamma": {
                    "init": [0.25, 0.25, 0.25, 0.25, 0.25]
                },
                "beta": {
                    "init": [-0.5, -0.5, -0.5, -0.5, -0.5]
                },
                "optimizer": {
                    "type": "QmlGradientDescent"
                },
                "penalty_weights": [1, 2.5, 2.5]
            }
        }


It is possible to further customize the :py:class:`QAOA<.QAOA>` with additional keyword arguments (see the QHyper API documentation). Below is presented an example of setting the `Pennylane simulator
type <https://pennylane.ai/plugins/>`_ using the ``backend`` keyword.

.. tabs::

    .. code-tab:: python

        from QHyper.solvers.gate_based.pennylane import QAOA
        from QHyper.optimizers import OptimizationParameter
        from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent

        solver = QAOA(problem,
            layers=5,
            gamma=OptimizationParameter(init=[0.25, 0.25, 0.25, 0.25, 0.25]),
            beta=OptimizationParameter(init=[-0.5, -0.5, -0.5, -0.5, -0.5]),
            optimizer=QmlGradientDescent(),
            backend="default.qubit",
            penalty_weights=[1, 2.5, 2.5],
        )


    .. code-tab:: yaml

        solver:
            category: gate_based
            platform: pennylane
            name: QAOA
            layers: 5
            gamma:
                init: [0.25, 0.25, 0.25, 0.25, 0.25]
            beta:
                init: [-0.5, -0.5, -0.5, -0.5, -0.5]
            optimizer: 
                type: QmlGradientDescent
            backend: default.qubit
            penalty_weights: [1, 2.5, 2.5]

    .. code-tab:: json

            {
                "solver": {
                    "category": "gate_based",
                    "platform": "pennylane",
                    "name": "QAOA",
                    "layers": 5,
                    "gamma": {
                        "init": [0.25, 0.25, 0.25, 0.25, 0.25]
                    },
                    "beta": {
                        "init": [-0.5, -0.5, -0.5, -0.5, -0.5]
                    },
                    "optimizer": {
                        "type": "QmlGradientDescent"
                    },
                    "backend": "default.qubit",
                    "penalty_weights": [1, 2.5, 2.5]
                }
            }



Customizing optimizers
^^^^^^^^^^^^^^^^^^^^^^

Customizing the ``optimizer`` settings is also possible. Below, a more detailed sample configuration is shown. Please note that adding all
native function options is possible (e.g., ``stepsize`` in this example is  native
from `Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_).

.. tabs::

    .. code-tab:: python

        from QHyper.solvers.gate_based.pennylane import QAOA
        from QHyper.optimizers import OptimizationParameter
        from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent

        solver = QAOA(problem,
            layers=5,
            gamma=OptimizationParameter(init=[0.25, 0.25, 0.25, 0.25, 0.25]),
            beta=OptimizationParameter(init=[-0.5, -0.5, -0.5, -0.5, -0.5]),
            optimizer=QmlGradientDescent(name='adam',
                                        steps=200,
                                        stepsize=0.005),
            penalty_weights=[1, 2.5, 2.5]
        )

    .. code-tab:: yaml

        solver:
            category: gate_based
            platform: pennylane
            name: QAOA
            layers: 5
            gamma:
                init: [0.25, 0.25, 0.25, 0.25, 0.25]
            beta:
                init: [-0.5, -0.5, -0.5, -0.5, -0.5]
            optimizer: 
                type: QmlGradientDescent
                name: adam
                steps: 200
                stepsize: 0.005
            backend: default.qubit
            penalty_weights: [1, 2.5, 2.5]

    .. code-tab:: json

        {
            "solver": {
                "category": "gate_based",
                "platform": "pennylane",
                "name": "QAOA",
                "layers": 5,
                "gamma": {
                        "init": [0.25, 0.25, 0.25, 0.25, 0.25]
                    },
                "beta": {
                    "init": [-0.5, -0.5, -0.5, -0.5, -0.5]
                },
                "optimizer": {
                    "type": "QmlGradientDescent",
                    "name": "adam",
                    "steps": 200,
                    "stepsize": 0.005
                },
                "backend": "default.qubit",
                "penalty_weights": [1, 2.5, 2.5]
            }
        }


Configuring a classical solver: Gurobi
--------------------------------------
.. tabs::

    .. code-tab:: python

        from QHyper.solvers.classical.gurobi import Gurobi

        solver = Gurobi(problem)

    .. code-tab:: yaml

        solver:
            category: classical
            platform: gurobi
            name: Gurobi

    .. code-tab:: json

        {
            "solver": {
                "category": "classical",
                "platform": "gurobi",
                "name": "Gurobi"
            }
        }



Combining optimizers and hyperoptimizers
----------------------------------------

It is also possible to make use of both the ``optimizer`` and the ``HyperOptimizer`` functionalities. The example below is similar to that in `Customizing optimizers`_. However, as in `Adding a hyperoptimizer`_, penalty weights  are searched by the ``HyperOptimizer`` within specified  bounds. In this example it is done using the Cross Entropy Search method (defined as :py:class:`cem<.CEM>`).  ``processes``, ``samples_per_epoch``, and ``epochs`` are parameters specific for ``CEM``.

.. note:: The `CEM` method is computationally expensive and may require a significant amount of time to complete (~5 min).


.. tabs::

    .. code-tab:: python

        from QHyper.solvers.gate_based.pennylane import WF_QAOA
        from QHyper.optimizers import OptimizationParameter
        from QHyper.optimizers.scipy_minimizer import ScipyOptimizer
        from QHyper.solvers.hyper_optimizer import HyperOptimizer
        from QHyper.optimizers.cem import CEM

        solver = WF_QAOA(problem,
            layers=5,
            gamma=OptimizationParameter(min=[0.0, 0.0, 0.0, 0.0, 0.0],
                                        init=[0.5, 0.5, 0.5, 0.5, 0.5],
                                        max=[6.28, 6.28, 6.28, 6.28, 6.28]),
            beta=OptimizationParameter(min=[0.0, 0.0, 0.0, 0.0, 0.0],
                                    init=[1.0, 1.0, 1.0, 1.0, 1.0],
                                    max=[6.28, 6.28, 6.28, 6.28, 6.28]),
            optimizer=ScipyOptimizer(),
            backend="default.qubit",
            penalty_weights=[1, 2.5, 2.5],
        )

        hyper_optimizer = HyperOptimizer(
            optimizer=CEM(processes=4,
                        samples_per_epoch=100,
                        epochs=5),
            solver=solver,
            penalty_weights={
                "min": [1, 1, 1],
                "max": [5, 5, 5],
                "init": [1, 2.5, 2.5]
            }
        )

    .. code-tab:: yaml

        solver:
            category: gate_based
            platform: pennylane
            name: WF_QAOA
            layers: 5
            gamma:
                min: [0, 0, 0, 0, 0]
                init: [0.5, 0.5, 0.5, 0.5, 0.5]
                max: [6.28, 6.28, 6.28, 6.28, 6.28]
            beta:
                min: [0, 0, 0, 0, 0]
                init: [1., 1., 1., 1., 1.]
                max: [6.28, 6.28, 6.28, 6.28, 6.28]
            optimizer: 
                type: scipy
            backend: default.qubit
        hyper_optimizer:
            optimizer: 
                type: cem
                processes: 4
                samples_per_epoch: 100
                epochs: 5
            penalty_weights: 
                min: [1, 1, 1]
                max: [5, 5, 5]
                init: [1, 2.5, 2.5]

    .. code-tab:: json

        {
            "solver": {
                "category": "gate_based",
                "platform": "pennylane",
                "name": "WF_QAOA",
                "layers": 5,
                "gamma": {
                    "min": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "init": [0.5, 0.5, 0.5, 0.5, 0.5],
                    "max": [6.28, 6.28, 6.28, 6.28, 6.28]
                },
                "beta": {
                    "min": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "init": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "max": [6.28, 6.28, 6.28, 6.28, 6.28]
                },
                "optimizer": {
                "type": "scipy"
                },
                "backend": "default.qubit"
            },
            "hyper_optimizer": {
                "optimizer": {
                    "type": "cem",
                    "processes": 4,
                    "samples_per_epoch": 100,
                    "epochs": 5
                },
                "penalty_weights": {
                    "min": [1, 1, 1],
                    "max": [5, 5, 5],
                    "init": [1, 2.5, 2.5]
                }
            }
        }



Supported optimizers
--------------------

A variety of (hyper)optimizers is supported. In QHyper the ``optimizer`` (both in a solver and in a hyperoptimizer)  can be set up using keyword arguments given below.

.. note::
    Please note that additional keyword arguments for each ``optimizer`` configuration can be taken directly from the native function definition (refer to the indicated  API documentation).

* :py:class:`.QmlGradientDescent`: customizable gradient descent set of optimizers from Pennylane (see below)
* :py:class:`.ScipyOptimizer`: `Scipy gradient descent set of optimizers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
* :py:class:`.Random`: Random optimizer (see QHyper API doc)
* :py:class:`.GridSearch`:  Grid search optimizer (see QHyper API doc)
* :py:class:`.CEM`: Cross Entropy Optimizer (see QHyper API doc)
* :py:class:`.Dummy`: Dummy optimizer (see QHyper API doc)

Additionally, the ``QmlGradientDescent`` set of optimizers can be further specified  (e.g. ``adam`` configuration was shown in point 6 above) using following keyword arguments (for details see `Pennylane documentation <https://docs.pennylane.ai/en/stable/introduction/interfaces.html#numpy>`_ ):

* ``adam``: qml.AdamOptimizer;
* ``adagrad``: qml.AdagradOptimizer;
* ``rmsprop``: qml.RMSPropOptimizer;
* ``momentum``: qml.MomentumOptimizer;
* ``nesterov_momentum``: qml.NesterovMomentumOptimizer;
* ``sgd``: qml.GradientDescentOptimizer;
* ``qng``: qml.QNGOptimizer.

Running solvers and hyperoptimizers
-----------------------------------

Running a pure solver:

.. tabs::

    .. code-tab:: py

        solver.solve()

    .. code-tab:: py Python using YAML

        # Note: the solver and problem configs should be in a single <file_name>.yaml file
        # ---
        #  solver:
        #         ...
        #  problem:
        #         ...

        import yaml
        from QHyper.solvers import solver_from_config

        with open("<file_name>.yaml", "r") as file:
            solver_config = yaml.safe_load(file)
        solver = solver_from_config(solver_config)
        solver.solve()


    .. code-tab:: py Python using JSON

        # Note: the solver and problem configs should be in a single <file_name>.json file
        # {
        #  "solver":
        #           { ... },
        #  "problem":
        #           { ... }
        # }

        import json
        from QHyper.solvers import solver_from_config

        with open("<file_name>.json", "r") as file:
            solver_config = json.load(file)
        solver = solver_from_config(solver_config)
        solver.solve()

Running a hyperoptimizer:

.. tabs::

    .. code-tab:: py

        hyper_optimizer.solve()
        hyper_optimizer.run_with_best_params()

    .. code-tab:: py Python using YAML

        # Note: the solver and problem configs should be in the same <file_name>.yaml file
        # ---
        #  solver:
        #         ...
        #  problem:
        #         ...

        import yaml
        from QHyper.solvers import solver_from_config

        with open("<file_name>.yaml", "r") as file:
            hyperoptimizer_config = yaml.safe_load(file)
        hyper_optimizer = solver_from_config(hyperoptimizer_config)
        hyper_optimizer.solve()
        hyper_optimizer.run_with_best_params()
        


    .. code-tab:: py Python using JSON

        # Note: the solver and problem configs should be in a single <file_name>.json file
        # {
        #  "solver":
        #           { ... },
        #  "problem":
        #           { ... }
        # }

        import json
        from QHyper.solvers import solver_from_config

        with open("<file_name>.json", "r") as file:
            hyper_optimizer_config = json.load(file)
        hyper_optimizer = solver_from_config(hyper_optimizer_config)
        hyper_optimizer.solve()
        hyper_optimizer.run_with_best_params()

You can explore how to evaluate the results by visiting the :doc:`demo/typical_use_cases` demo.
