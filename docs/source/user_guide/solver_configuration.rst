=============================
Solver configuration tutorial
=============================

Problem definition
------------------

This tutorial assumes following sample optimization problem definition:

.. code-block:: yaml

    problem:
        type: knapsack
        max_weight: 2
        items_weights: [1, 1, 1]
        items_costs: [2, 2, 1]

Which defines `knapsack problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_ of filling a knapsack with three items, each characterized with weight and cost.
The goal is to put chosen items in the knapsack to achieve maximal cost  with total weight not exceeding  ``max_weight``


Solver types
------------

Basic solver definition requires providing its type. Currently supported types are:

* ``vga`` for variational algorithms
* ``cqm`` for `D-Wave CQM <https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html#leap-s-hybrid-solvers>`_
* ``dqm`` for `D-Wave DQM <https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html#leap-s-hybrid-solvers>`_
* ``advantage`` for `D-Wave Advantage <https://docs.dwavesys.com/docs/latest/c_gs_4.html>`_ (currently default advantage_system5.4. is supported)
* ``gurobi`` for `Gurobi Optimizer <https://www.gurobi.com/solutions/gurobi-optimizer/>`_

Sample code for defining type advantage solver

.. code-block:: yaml

    solver:
        type: advantage


Configuring initial QUBO penalties (Lagrangian multipliers)
-----------------------------------------------------------

``advantage`` solver requires problem definition in the `QUBO <https://arxiv.org/pdf/1811.11538>`_ form. QHyper automatically creates the QUBO for
knapsack problem 

.. math::
   f(\boldsymbol{x}, \boldsymbol{y}) = 
   - \alpha_0 \underbrace{\sum_{i = 1}^N c_i x_i}_{\text{cost function}} + \alpha_1 \underbrace{(1 - \sum_{i=1}^W y_i)^2}_{\text{coding constraint}} + \alpha_1 \underbrace{(\sum_{i=1}^W iy_i - \sum_{i=1}^N w_ix_i)^2}_{\text{weight constraint}},
   
where 
 * :math:`N=3` is the number of items available, 
 * :math:`W=` ``max_weight`` is the maximum weight of the knapsack, 
 * :math:`c_i` and :math:`w_i` are the costs and weights specified in ``items_costs`` and ``items_weights`` lists of the configuration. 
 * The goal is to optimize :math:`\boldsymbol{x} = [x_i]_N` which is a Boolean vector, where :math:`x_i = 1`  if and only if the item :math:`i` was selected to be inserted into the knapsack. 
 * :math:`\boldsymbol{y} = [y_i]_W` is a one-hot vector where :math:`y_i = 1` if and only if the weight of the knapsack is equal to :math:`i`; 
 * :math:`\alpha_j` are penalty weights  (i.e. Lagrangian multipliers, hyperparameters of the optimized function).

Therefore, the proper function definition  requires setting  three :math:`\alpha_j` penalties  i.e. hyperparameters
for the cost function and two constraints: ensuring that problem encoding is correct and that knapsack weight fullfils 
``max_weight`` requirement .

In the example below, the constraint penalties  are set as ``hyper_args``

.. code-block:: yaml

    solver:
        type: advantage
        params_inits:
            hyper_args: [1, 2.5, 2.5]

Adding hyperoptimizer
---------------------

Since guessing correct penalties is often difficult, there is also option to define ``hyper_optimizer`` to search for appropriate settings.
In the example below, ``grid`` search hyperoptimizer is applied to find  proper penalties  of the  knapsack optimized function.
The penalties are searched within specified  ``bounds`` with ``steps`` defined in the configuration.

.. code-block:: yaml

    solver:
        type: advantage
        hyper_optimizer:
            type: grid
            steps: [0.01, 0.01, 0.01]
            bounds: [[1, 10], [1, 10], [1, 10]]

Configuring variational quantum algorithms
------------------------------------------

``vqa`` solver type is a  set containing solvers based on  gate-based variational algorithms. Currenly `QAOA <https://arxiv.org/abs/1411.4028>`_, `WF-QAOA and H-QAOA <https://www.iccs-meeting.org/archive/iccs2023/papers/140770117.pdf>`_
are supported obtained by setting ``pqc type`` to  ``qaoa``, ``wfqaoa`` and ``hqaoa`` repectively.

Typical example of QAOA configuration is shown below. The parametrized quantum circuit is configured for  5 ``layers``.  Default local
`Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_ ``optimizer``
from `Pennylane <https://pennylane.ai/>`_ (``type: qml``) with default options is used.

Initial variational parameters optimized by Adam method are set as ``angles``.   Penalty weights are initialized  as ``hyper_args``.

.. code-block:: yaml

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


It is possible to further customized ``pqc`` with additional keyword arguments (see QHyper API documentation). Below example of setting `Pennylane simulator
type <https://pennylane.ai/plugins/>`_ for ``qaoa``  using ``backend`` keyword

.. code-block:: yaml

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


Customizing optimizers
----------------------

Customising ``optimizer`` settings is also possible. Below, more detailed sample configuration is shown. Please note that adding all
native function options is possible (e.g. ``stepsize`` in this example are native
from `Adam gradient  descent <https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html>`_   )

.. code-block:: yaml

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


Combining optimizers and hyperoptimizers
----------------------------------------

It is also possible to make use of both ``optimizer`` and ``hyper_optimizer`` functionality. The example below is similar to that in `Customizing optimizers`_.
However, as in `Adding hyperoptimizer`_, penalties  are searched by ``hiper_optimizer`` within specified  ``bounds``. In this example it is done  by Cross Entropy Search  method (configured as ``cem``).  ``processes``, ``samples_per_epoch`` and ``epochs`` are parameters specific for ``cem``.

.. code-block:: yaml

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


Supported optimizers
--------------------

Variety of (hyper)optimizers. In QHyper both ``hyper_optimizer`` and ``optimizer`` can be set up using keyword arguments given below.

.. note::
    Please note that additional keyword arguments for each ``optimizer`` or ``hyper_optimizer`` configuration can be taken directly from native  function definition (refer to indicated  API documentation).

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
