Defining custom problems, optimizers and solvers
================================================

Big advantage of using QHyper is the ability to run experiments from configuration file.
But this only allows to use predefined problems, optimizers and solvers.
To overcome this limitation, QHyper will try to import any :py:class:`.Problem`, :py:class:`.Optimizer` or :py:class:`.Solver` class from directory `custom` or `QHyper/custom`.
It is required that these classes inherit from one of them and implement required methods.
