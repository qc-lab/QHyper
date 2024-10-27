=============================
Solver configuration tutorial
=============================

Problem definition
------------------

This tutorial assumes the following sample optimization problem definition:

.. tabs::

    .. code-tab:: yaml

        problem:
            type: knapsack
            max_weight: 2
            items_weights: [1, 1, 1]
            items_values: [2, 2, 1]

    .. code-tab:: json

        {
            "problem": {
                "type": "knapsack",
                "max_weight": 2,
                "items_weights": [1, 1, 1],
                "items_values": [2, 2, 1]
            }
        }


Which defines the `Knapsack Problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_ of filling a knapsack with three items, each characterized with weight and cost.
The goal is to put the selected items in the knapsack to achieve maximal cost with total weight not exceeding the ``max_weight``.


Solver types
------------

Basic solver definition requires providing its type. Currently supported types are:

* :py:class:`vqa<.VQA>` for variational algorithms
* :py:class:`cqm<.CQM>` for `D-Wave CQM <https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html#leap-s-hybrid-solvers>`_
* :py:class:`dqm<.DQM>` for `D-Wave DQM <https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html#leap-s-hybrid-solvers>`_
* :py:class:`advantage<.Advantage>` for `D-Wave Advantage <https://docs.dwavesys.com/docs/latest/c_gs_4.html>`_ (currently default advantage_system5.4. is supported)
* :py:class:`gurobi<.Gurobi>` for `Gurobi Optimizer <https://www.gurobi.com/solutions/gurobi-optimizer/>`_

Sample code for defining type advantage solver

.. tabs::

        .. code-tab:: yaml

            solver:
                type: advantage

        .. code-tab:: json

            {
                "solver": {
                    "type": "advantage"
                }
            }


Configuring the initial QUBO penalties (Lagrangian multipliers)
---------------------------------------------------------------

The ``advantage`` solver requires problem definition in the `QUBO <https://arxiv.org/pdf/1811.11538>`_ form. QHyper automatically creates the QUBO for
the Knapsack Problem

.. math::
   f(\boldsymbol{x}, \boldsymbol{y}) =
   - \alpha_0 \underbrace{\sum_{i = 1}^N c_i x_i}_{\text{cost function}} + \alpha_1 \underbrace{(1 - \sum_{i=1}^W y_i)^2}_{\text{constraint encoding}} + \alpha_2 \underbrace{(\sum_{i=1}^W iy_i - \sum_{i=1}^N w_ix_i)^2}_{\text{weight constraint}},

where
 * :math:`N=3` is the number of available items;
 * :math:`W=` ``max_weight`` is the maximum weight of the knapsack;
 * :math:`c_i` and :math:`w_i` are the costs and weights specified in ``items_values`` and ``items_weights`` lists of the configuration;
 * The goal is to optimize :math:`\boldsymbol{x} = [x_i]_N` which is a Boolean vector, where :math:`x_i = 1`  if and only if the item :math:`i` was selected to be inserted into the knapsack;
 * :math:`\boldsymbol{y} = [y_i]_W` is a one-hot vector where :math:`y_i = 1` if and only if the weight of the knapsack is equal to :math:`i`;
 * :math:`\alpha_j` are penalty weights  (i.e. Lagrangian multipliers, hyperparameters of the optimized function).

Therefore, the proper function definition  requires setting  the three :math:`\alpha_j` penalties  i.e. hyperparameters
for the cost function and two constraints: ensuring that problem encoding is correct and that knapsack weight fullfils
``max_weight`` requirement .

In the example below, the constraint penalties  are set as ``hyper_args``

.. tabs::

    .. code-tab:: yaml

        solver:
            type: advantage
            params_inits:
                hyper_args: [1, 2.5, 2.5]

    .. code-tab:: json

        {
            "solver": {
                "type": "advantage",
                "params_inits": {
                    "hyper_args": [1, 2.5, 2.5]
                }
            }
        }


Adding the hyperoptimizer
-------------------------

Since guessing the correct penalties is often a difficult task, there is also an option to define the ``hyper_optimizer`` to search for the appropriate settings.
In the example below, :py:class:`grid<.GridSearch>` search hyperoptimizer is applied to find  proper penalties  of the  knapsack optimized function.
The penalties are searched within specified  ``bounds`` with ``steps`` defined in the configuration.

.. tabs::

    .. code-tab:: yaml

        solver:
            type: advantage
            hyper_optimizer:
                type: grid
                steps: [0.01, 0.01, 0.01]
                bounds: [[1, 10], [1, 10], [1, 10]]

    .. code-tab:: json

        {
            "solver": {
                "type": "advantage",
                "hyper_optimizer": {
                    "type": "grid",
                    "steps": [0.01, 0.01, 0.01],
                    "bounds": [[1, 10], [1, 10], [1, 10]]
                }
            }
        }

Configuring variational quantum algorithms
------------------------------------------

:py:class:`.VQA` solver type is a base class containing solvers for gate-based variational algorithms. Currenly `QAOA <https://arxiv.org/abs/1411.4028>`_, `WF-QAOA and H-QAOA <https://www.iccs-meeting.org/archive/iccs2023/papers/140770117.pdf>`_
are supported and can be used by setting ``pqc.type`` to :py:class:`qaoa<.QAOA>`, :py:class:`wfqaoa<.WFQAOA>`, and :py:class:`hqaoa<.HQAOA>` repectively.

A typical example of the QAOA configuration is presented below. The parameterized quantum circuit is configured for  5 ``layers``.  Default local
`Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_ ``optimizer``
from `Pennylane <https://pennylane.ai/>`_ (``type:`` :py:class:`qml<.QmlGradientDescent>`) with default options is used.

Initial variational parameters optimized by the Adam method are set as ``angles``. Penalty weights are initialized as ``hyper_args``.

.. tabs::

    .. code-tab:: yaml

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

    .. code-tab:: json

        {
            "solver": {
                "type": "vqa",
                "pqc": {
                    "type": "qaoa",
                    "layers": 5
                },
                "optimizer": {
                    "type": "qml"
                },
                "params_inits": {
                    "angles": [[0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1]],
                    "hyper_args": [1, 2.5, 2.5]
                }
            }
        }


It is possible to further customize :py:class:`pqc<.PQC>` with additional keyword arguments (see the QHyper API documentation). Below is presented an example of setting `Pennylane simulator
type <https://pennylane.ai/plugins/>`_ for :py:class:`qaoa<.QAOA>` using the ``backend`` keyword

.. tabs::

    .. code-tab:: yaml

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

    .. code-tab:: json

        {
            "solver": {
                "type": "vqa",
                "pqc": {
                    "type": "qaoa",
                    "layers": 5,
                    "backend": "default.qubit"
                },
                "optimizer": {
                    "type": "qml"
                },
                "params_inits": {
                    "angles": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1, 1]
                    ],
                    "hyper_args": [1, 2.5, 2.5]
                }
            }
        }



Customizing optimizers
----------------------

Customising the ``optimizer`` settings is also possible. Below, a more detailed sample configuration is shown. Please note that adding all
native function options is possible (e.g., ``stepsize`` in this example is  native
from `Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_   )

.. tabs::

    .. code-tab:: yaml

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

    .. code-tab:: json

        {
            "solver": {
                "type": "vqa",
                "pqc": {
                    "type": "qaoa",
                    "layers": 5
                },
                "optimizer": {
                    "type": "qml",
                    "optimizer": "adam",
                    "steps": 200,
                    "stepsize": 0.005
                },
                "params_inits": {
                    "angles": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1, 1]
                    ],
                    "hyper_args": [1, 2.5, 2.5]
                }
            }
        }



Combining optimizers and hyperoptimizers
----------------------------------------

It is also possible to make use of both the ``optimizer`` and the ``hyper_optimizer`` functionalities. The example below is similar to that in `Customizing optimizers`_.
However, as in `Adding the hyperoptimizer`_, penalties  are searched by the ``hyper_optimizer`` within specified  ``bounds``. In this example it is done by the Cross Entropy Search method (configured as :py:class:`cem<.CEM>`).  ``processes``, ``samples_per_epoch``, and ``epochs`` are parameters specific for ``cem``.

.. tabs::

    .. code-tab:: yaml

        solver:
        type: vqa
        pqc:
            type: wfqaoa
            layers: 5
        optimizer:
            type: qml
            optimizer: adam
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

    .. code-tab:: json

        {
            "solver": {
                "type": "vqa",
                "pqc": {
                    "type": "wfqaoa",
                    "layers": 5
                },
                "optimizer": {
                    "type": "qml",
                    "optimizer": "adam",
                    "steps": 200,
                    "stepsize": 0.005
                },
                "hyper_optimizer": {
                    "type": "cem",
                    "processes": 4,
                    "samples_per_epoch": 1000,
                    "epochs": 10,
                    "bounds": [
                        [1, 10],
                        [1, 10],
                        [1, 10]
                    ]
                },
                "params_inits": {
                    "angles": [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1, 1]
                    ],
                    "hyper_args": [1, 2.5, 2.5]
                }
            }
        }



Supported optimizers
--------------------

Aariety of (hyper)optimizers is supported. In QHyper both the ``hyper_optimizer`` and the ``optimizer`` can be set up using keyword arguments given below.

.. note::
    Please note that additional keyword arguments for each ``optimizer`` or ``hyper_optimizer`` configuration can be taken directly from the native function definition (refer to the indicated  API documentation).

* :py:class:`.QmlGradientDescent`: customizable gradient descent set of optimizers from Pennylane  (see below)
* :py:class:`.ScipyOptimizer`: `Scipy gradient descent set of optimizers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
* :py:class:`.Random`: Random optimizer (see QHyper API doc)
* :py:class:`.GridSearch`:  Grid search optimizer (see QHyper API doc)
* :py:class:`.CEM`: Cross Entropy Optimizer (see QHyper API doc)
* :py:class:`.Dummy`: Dummy optimizer (see QHyper API doc)

Additionally, the ``qml`` set of optimizers can be further specified  (e.g. ``adam`` configuration was shown in point 6 above) using following keyword arguments (for details see `Pennylane documentation <https://docs.pennylane.ai/en/stable/introduction/interfaces.html#numpy>`_ ):

* ``adam``: qml.AdamOptimizer;
* ``adagrad``: qml.AdagradOptimizer;
* ``rmsprop``: qml.RMSPropOptimizer;
* ``momentum``: qml.MomentumOptimizer;
* ``nesterov_momentum``: qml.NesterovMomentumOptimizer;
* ``sgd``: qml.GradientDescentOptimizer;
* ``qng``: qml.QNGOptimizer.
