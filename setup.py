from setuptools import setup, find_packages


DESCRIPTION = 'Quantum and classical combinatorial optimization problem solvers'
LONG_DESCRIPTION = 'A package that allows to build and solve quantum and classical problems using predefined solvers and problems.'


# Setting up
setup(
    name="qhyper",
    version='0.4.0',
    author="ACC Cyfronet AGH",
    author_email="jzawalska@agh.edu.pl",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.12',
    install_requires=[
        "numpy~=2.2.6",
        "PennyLane~=0.41.1",
        "pennylane-lightning~=0.41.1",
        "autoray~=0.7.2",
        "tqdm~=4.67.3",
        "sympy~=1.13.0",
        "dwave-system~=1.36.0",
        "dwave-cloud-client~=0.14.8",
        "gurobipy~=12.0.0",
        "pandas~=2.2.3",
        "qiskit~=2.1.2",
        "qiskit-aer~=0.17.2",
        "iqm-client~=34.0.4",
        "py4lexis @ git+https://opencode.it4i.eu/lexis-platform/clients/py4lexis.git",
        "qaas @ git+https://github.com/It4innovations/quantum-as-a-service.git",
        "wfcommons @ git+https://github.com/wfcommons/wfcommons.git@main",
    ],
    extras_require={
        "dev": [
            "types-tqdm~=4.67.3.20260205",
            "ipywidgets~=8.1",
        ],
    },
    keywords=['python', 'qhyper', 'quantum', 'solver', 'experiment'],
    license='MIT',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
