from setuptools import setup, find_packages


DESCRIPTION = 'Quantum and classical problem solvers'
LONG_DESCRIPTION = 'A package that allows to build and solve quantum and classical problems using predefined solvers and problems.'


# Setting up
setup(
    name="qhyper",
    version='0.3.4',
    author="ACC Cyfronet AGH",
    author_email="jzawalska@agh.edu.pl",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy~=1.26.4', 'PennyLane~=0.38', 'tdqm', 'sympy==1.13.1', 'dwave-system', 'gurobipy', 'pandas', 'wfcommons'],
    keywords=['python', 'qhyper', 'quantum', 'solver', 'experiment'],
    license='MIT',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
