from setuptools import setup, find_packages
import codecs
import os
import re
import subprocess


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


def is_valid_version(tag: str) -> bool:
    # Basic PEP 440 version regex (supporting versions like 1.0.0, 1.0, 1.0a1, 1.0.dev1, etc.)
    pattern = r"^\d+(\.\d+){0,2}([a-z]+\d+)?(\.post\d+)?(\.dev\d+)?$"
    return bool(re.match(pattern, tag))


def get_version():
    try:
        tag = subprocess.check_output(["git", "describe", "--tags"]).strip().decode("utf-8")
        tag = tag.replace('test/', '')
        has_v_prefix = tag.startswith('v')
        version_core = tag[1:] if has_v_prefix else tag
        final_version = None
        if is_valid_version(version_core):
            final_version = f"v{version_core}" if has_v_prefix else version_core
        else:
            git_describe_pattern = r"^(\d+\.\d+\.\d+)-(\d+)-g[0-9a-f]+$"
            match = re.match(git_describe_pattern, version_core)
            if match:
                base_version, commit_count = match.groups()
                fixed_version = f"{base_version}.post{commit_count}"
                final_version = f"v{fixed_version}" if has_v_prefix else fixed_version

            raise ValueError(f"Cannot automatically fix version '{tag}'")
        return final_version
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
