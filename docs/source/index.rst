.. documentation master file, created by
   sphinx-quickstart on Thu Oct 27 13:27:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
    :google-site-verification: -BAN3UWgNz2CPUt5v5AcQpDH8GJc0kX0VdKh2Kfj89I


Welcome to QHyper's documentation!
==================================

QHyper - a software framework for hybrid quantum-classical combinatorial optimization.

Introduction to QHyper Library
==============================

Welcome to the documentation for the **QHyper** library, a powerful tool designed
to simplify the process of conducting combinatorial optimization experiments in the realm of quantum
algorithms. The primary aim of QHyper is to provide a
user-friendly framework that simplifies the execution of experiments, focusing on
quantum algorithms while also allowing classical approaches.

Key Features
------------

- **Quantum Algorithm Emphasis:** QHyper is tailored for researchers and
  developers interested in exploring quantum combinatorial optimization algorithms. It offers an environment
  that facilitates the implementation of quantum-based solvers.

- **Classical Solver Support:** While the focus is on quantum algorithms,
  QHyper acknowledges the significance of classical methods. It enables the
  seamless integration of classical solvers, ensuring a unified platform for
  comparative experiments.

- **Modularity and Extensibility:** One of QHyper's core strengths is its design
  for easy extensibility. Adding new solvers, algorithms, or problems is
  straightforward, empowering users to contribute and customize the library to
  suit their research needs.

- **Simplified Experimentation:** With QHyper, the experimentation process is
  made more accessible. Researchers can define, execute, and analyze experiments
  efficiently, allowing them to focus on the insights derived from the results.

- **Hyperparameters optimization:**  QHyper offers set of hyperoptimizers with easy configuration when proper formulation of a constrained optimization problem is needed by quantum  solver accepting unconstrained forms only.


Architecture
================

QHyper consists with three main components:

* :doc:`Problems <problems>` - classes describing different types of problems e.g. Knapsack problem or Travelling Salesman Problem

* :doc:`Solvers <solvers>` - classes defining different types of solvers e.g. quantum solvers like QAOA or CQM, but also classical algorithms like Gurobi

* :doc:`Optimizers <optimizers>` - classes implementing different types of optimizer

Each abstract class allows adding new implementatons which will be compatible with the rest of the system.

.. raw:: html

      <img src="_static/qhyper_architecture.svg" style="background-color: transparent">


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
