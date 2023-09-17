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
        }
    }
    vqa = Solver.from_config(problem, config=solver_config)


5. **Execute solver:** Run your experiments using the solver on defined problem. Analyze the outcomes and gather insights from the results.
    
.. code-block:: python

    vqa.solve(params_cofing)
    # {
    #     'angles': array([0.20700558, 0.85908389, 0.58705246, 0.52192666, 0.7343595 ,
    #                      0.98882406, 1.00509525, 1.0062573 , 0.9811152 , 1.12500301]),
    #     'hyper_args': array([1.02005268, 2.10821942, 1.94148846, 2.14637279, 1.88565744])
    # }


Conclusion
----------

Congratulations! You've just scratched the surface of what the `QHyper` library
can offer. By following this guide, you've learned how to install the library,
embrace quantum algorithm and set up your initial
experiments.

For more advanced usage and examples, explore the
`demos <https://github.com/qc-lab/QHyper/tree/main/demo>` available on github.

Happy experimenting with `QHyper`!
