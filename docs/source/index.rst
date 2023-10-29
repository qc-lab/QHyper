.. documentation master file, created by
   sphinx-quickstart on Thu Oct 27 13:27:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
    :google-site-verification: -BAN3UWgNz2CPUt5v5AcQpDH8GJc0kX0VdKh2Kfj89I


Welcome to QHyper's documentation!
==================================

.. image:: https://user-images.githubusercontent.com/38388283/226841016-711112a8-09d1-4a83-8aab-6e305cb24edb.png
    :width: 40

QHyper - a software framework for hybrid quantum-classical optimization.

Introduction to QHyper Library
==============================

Welcome to the documentation for the **QHyper** library, a powerful tool designed
to simplify the process of conducting experiments in the realm of quantum
algorithms. The primary aim of QHyper is to provide a
user-friendly framework that simplifies the execution of experiments, focusing on
quantum algorithms while also allowing classical approaches.

Key Features
------------

- **Quantum Algorithm Emphasis:** QHyper is tailored for researchers and
  developers interested in exploring quantum algorithms. It offers an environment
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


Architecture
================

QHyper consists with three main components:
    * :ref:`README Problems` - classes describing different types of problems e.g. Knapsack problem or Travelling Salesman Problem
    * :ref:`README Solvers` - classes definng differnt types of solvers e.g. quantum solvers like QAOA or CQM, but also classical algorithms like Gurobi
    * :ref:`README Optimizers` - classes implementing different types of optimizer

Each abstract class allows adding new implementatons which will be compatible with the rest of the system.

.. _README Problems:

Problems
----------

.. graphviz:: _static/classes_problems.dot

More detailed information can be found on :ref:`generated/qhyper.problems.base.problem:qhyper.problems.base.problem`.


.. _README Solvers:

Solvers
----------

.. graphviz:: _static/classes_solvers.dot


More detailed information can be found on :ref:`generated/qhyper.solvers.base.solver:qhyper.solvers.base.solver`.

.. _README Optimizers:

Optimizers
----------

.. graphviz:: _static/classes_optimizers.dot


More detailed information can be found on :ref:`generated/qhyper.optimizers.base.optimizer:qhyper.optimizers.base.optimizer`.



Contents
--------
.. toctree::

   Home <self>
   usage
   contribution
   api
   changelog


