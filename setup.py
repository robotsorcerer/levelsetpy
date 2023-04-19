from setuptools import setup, find_packages

ver = {}
try:
    with open('Utilities/_version.py') as fd:
        exec(fd.read(), ver)
    version = ver.get('__version__', '0.0.dev0')
except IOError:
    version = 'dev'

with open('README.md') as fp:
    long_description = fp.read()

CLASSIFIERS = """
Development Status :: 3 - Alpha
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


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='levelsetpy',
    version=version,
    author='Lekan Molu',
    author_email='lekanmolu@microsoft.com',
    url='https://github.com/robotsorcerer/LevelSetPy',
    description='GPU-Accelerated Level Set Methods in Python',
    long_description=long_description,
    packages=find_packages(),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    # install_requires=['numpy',
    #                   'scipy',
    #                   'cupy==11.3',
    #                   'absl-py',
    #                   'scikit-image',
    #                   'matplotlib'],
    install_requires=required,
)
