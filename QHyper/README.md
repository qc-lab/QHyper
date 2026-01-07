
<p align="center">
    <img width="100px" alt="qhyper_logo" src="docs/source/_static/logo.png">
</p>

QHyper is a Python library that provides a unified interface for experimenting with quantum-related optimization solvers. It allows users to specify combinatorial optimization problems, select solvers, manage problem hyperparameters, and standardize output for ease of use.

## Introduction

QHyper is a library, which is aimed at users working on
computational experiments with a variety of quantum combinatorial optimization solvers.
The library offers a simple and extensible interface for formulating combinatorial
optimization problems, selecting and running solvers,
and optimizing hyperparameters. The supported solver set includes variational
gate-based algorithms, quantum annealers, and classical solutions.
The solvers can be combined with provided local and global (hyper)optimizers.
The main features of the library are its extensibility on different levels of use
as well as a straightforward and flexible experiment configuration format.

## Documentation

The latest documentation can be found on [Readthedocs](https://qhyper.readthedocs.io/en/latest/).

## Installation

To install QHyper, use the following command (ensure that you have Python 3.12 installed before running the command).

``` bash
pip install qhyper
```
The latest version of QHyper can be downloaded and installed directly from github. But please be careful, this version may not be released yet and may contain bugs.
```bash
pip install git+https://github.com/qc-lab/QHyper
```


## Key features

- **Quantum algorithm emphasis:** QHyper is designed for researchers and developers exploring quantum optimization algorithms, providing an environment for implementing quantum and hybrid quantum-classical solvers.

- **Classical solver support:** While the focus is on quantum algorithms, QHyper also enables seamless integration of classical solvers, ensuring a unified platform for comparative experiments.

- **Simplified experimentation:** With QHyper, the experimentation process is made more accessible. The users can define, execute, and analyze experiments efficiently due to the unified formats of inputs and outputs, and possibility of using configuration files.

- **Modularity and extensibility:** One of QHyper's core strengths is easy extensibility. Adding new problems, solvers, or optimizers is straightforward, empowering users to contribute and customize the library to suit their research needs.

- **Hyperparameters optimization:** QHyper offers easily configurable hyperoptimizers for converting constrained optimization problems into unconstrained forms required by some quantum solvers.


## Architecture

The architecture of QHyper is presented on a diagram below:

<img src="docs/source/_static/qhyper_architecture.svg" style="background-color: transparent">

The main components are:

* [Problems](https://qhyper.readthedocs.io/en/latest/generated/QHyper.problems.html) - classes that describe different types of problems, such as the Knapsack Problem or the Traveling Salesman Problem.

* [Solvers](https://qhyper.readthedocs.io/en/latest/generated/QHyper.solvers.html) - Classes that define different types of solvers, e.g., quantum/hybrid solvers like the Quantum Approximate Optimization Algorithm or the Constrained Quadratic Model, but also classical solvers like Gurobi.

* [Optimizers](https://qhyper.readthedocs.io/en/latest/generated/QHyper.optimizers.html) - Classes that implement different types of (hyper)optimizers.

Each abstract class allows adding new implementatons which will be compatible with the rest of the system.

