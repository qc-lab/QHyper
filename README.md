# Content
- [CEM.pdf](CEM.pdf) - description of CEM algorithm and explanation how to calculate parameters for next epoch
- [CEM.ipynb](CEM.ipynb) - notebook with CEM implementation for optimising 2d function (only classical calculation)
- [QAOA_maxcut.ipynb](QAOA_maxcut.ipynb) - notebook with CEM implementation for optimising max-cut problem. Actual use case of optimization algorithm for finding best results using quantum computers.
- Knapsack problem:
    - [QAOA_knapsack.ipynb](QAOA_knapsack.ipynb) - notebook with CEM implementation for optimising knapsack problems. Doesn't work because of the problems related to Qiskit QAOA.
    - [CEM_optimizer.py](CEM_optimizer.py) - implementation of custom Qiskit Optimizer based on CEM. Unfortunately it doesn't work with Qiskit QAOA.
