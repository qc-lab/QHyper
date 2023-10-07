from setuptools import setup, find_packages
import codecs
import os
import subprocess
import re


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

def get_version():
    try:
        tag = subprocess.check_output(["git", "describe", "--tags"]).strip().decode("utf-8")
        tag = tag.replace('test/', '')
        return tag
    except Exception as e:
        print("Error:", e)
        return "0.dev1"  # Default version if Git tag extraction fails


DESCRIPTION = 'Quantum and classical problem solvers'
LONG_DESCRIPTION = 'A package that allows to build and solve quantum and classical problems using predefined solvers and problems.'


# Setting up
setup(
    name="qhyper",
    version=get_version(),
    author="ACK Cyfronet AGH",
    author_email="tomasz.lamza@cyfronet.pl",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'PennyLane', 'tdqm', 'sympy', 'dwave-system', 'gurobipy', 'pandas', 'wfcommons'],
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
