#!/usr/bin/env python

# This future is needed to print Python2 EOL message
from __future__ import print_function
import sys
if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is not supported by LevelSetPy.")
    sys.exit(-1)
if sys.platform == 'win32' and sys.maxsize.bit_length() == 31:
    print("32-bit Windows Python runtime is not supported. Please switch to 64-bit Python.")
    sys.exit(-1)

import platform
python_min_version = (3, 7, 0)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(platform.python_version(),
                                                                     python_min_version_str))
    sys.exit(-1)


from setuptools import setup, find_packages
# from distutils.core import setup


CLASSIFIERS = """
Development Status :: 0 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

excludes=["Tests", "Figures"]

with open('README.md') as fp:
    long_description = fp.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='LevelSetPy',
    version='0.0',
    author='Lekan Molu',
    author_email='lekanmolu@microsoft.com',
    url='https://github.com/robotsorcerer/LevelSetPy',
    description='GPU-Accelerated Hyperbolic PDE Solvers and Level Set Dynamics to Implicit Geometries in Python',
    long_description=long_description,
    packages=['LevelSetPy', 'LevelSetPy.BoundaryCondition', 'LevelSetPy.DynamicalSystems', 'LevelSetPy.ExplicitIntegration', \
                'LevelSetPy.Grids', 'LevelSetPy.InitialConditions', 'LevelSetPy.SpatialDerivative', \
                'LevelSetPy.Utilities', 'LevelSetPy.Visualization'],
    # packages=find_packages(exclude=excludes),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    install_requires=required,
)
