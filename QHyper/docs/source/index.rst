.. documentation master file, created by
   sphinx-quickstart on Thu Oct 27 13:27:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
    :google-site-verification: -BAN3UWgNz2CPUt5v5AcQpDH8GJc0kX0VdKh2Kfj89I


QHyper documentation
==================================

| QHyper is a Python library that provides a unified interface for experimenting with quantum-related optimization solvers. 
| It allows users to specify combinatorial optimization problems, select solvers, manage problem hyperparameters, and standardize output for ease of use.

Installation
^^^^^^^^^^^^
To install QHyper, use the following command (ensure that you have Python 3.12 installed before running the command).

.. code-block:: bash

    pip install qhyper


Key features
------------

- **Quantum algorithm emphasis:** QHyper is designed for researchers and developers exploring quantum optimization algorithms, providing an environment for implementing quantum and hybrid quantum-classical solvers.

- **Classical solver support:** While the focus is on quantum algorithms, QHyper also enables seamless integration of classical solvers, ensuring a unified platform for comparative experiments.

- **Simplified experimentation:** With QHyper, the experimentation process is made more accessible. The users can define, execute, and analyze experiments efficiently due to the unified formats of inputs and outputs, and possibility of using configuration files.

- **Modularity and extensibility:** One of QHyper's core strengths is easy extensibility. Adding new problems, solvers, or optimizers is straightforward, empowering users to contribute and customize the library to suit their research needs.

- **Hyperparameters optimization:** QHyper offers easily configurable hyperoptimizers for converting constrained optimization problems into unconstrained forms required by some quantum solvers.


Architecture
------------

The architecture of QHyper is presented on a diagram below:

.. raw:: html

      <img src="_static/qhyper_architecture.svg" style="background-color: transparent">

The main components are:

* :doc:`Problems <generated/QHyper.problems>` - classes that describe different types of problems, such as the Knapsack Problem or the Traveling Salesman Problem.

* :doc:`Solvers <generated/QHyper.solvers>` - Classes that define different types of solvers, e.g., quantum/hybrid solvers like the Quantum Approximate Optimization Algorithm or the Constrained Quadratic Model, but also classical solvers like Gurobi.

* :doc:`Optimizers <generated/QHyper.optimizers>` - Classes that implement different types of (hyper)optimizers.

Each abstract class allows adding new implementatons which will be compatible with the rest of the system.


.. toctree::
    :hidden:

    Home <self>

.. toctree::
    :hidden:

    user_guide/index

.. toctree::
    :hidden:

    contribution

.. toctree::
   :hidden:

   api