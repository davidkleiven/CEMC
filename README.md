![Build status](https://travis-ci.org/davidkleiven/WangLandau.svg?branch=master)
# WangLandau

# Dependencies
* [SWIG](http://www.swig.org/) (newer than version 3.0)
* [GCC](https://gcc.gnu.org/) (has to support multithreading)
* [MPI](https://www.mpich.org/)
* Python packages listed in [requirements.txt](requirements.txt)

# Installation
Install all the dependencies on Ubuntu
```bash
sudo apt-get update
sudo apt-get install swig
sudo apt-get install g++
sudo apt-get install mpich2
```

Install the python dependencies (at the moment this only works for Python 2)
```bash
pip install -r requirements.txt
```

Install the package
```bash
pip install .
```

# Examples
Examples of application of the package are listed below

* [Ground State of Al-Mg](examples/test_ground_state.ipynb)
* [Monte Carlo in SGC Ensemble](examples/test_sgc_mc.ipynb)

# Troubleshooting
1. **Missing C++ version of CE updater** try to install with
```bash
pip install -e .
```
instead.

2. **Equations does not render properly when viewing the examples**
try to convert the jupyter notebook locally
```bash
jupyter nbconvert --to html test_sgc_mc.ipynb
```
and open the resulting html file in a browser.

# Guidelines
Any code that is in this repository should have *at least* one unittest
located in the *tests* folder. The minimum test required is that the
code does what it is supposed to do without raising unexpected exceptions.
Code producing results that can be verified against some reference values,
should also include tests verifying that the code produce the desired result.

All examples should be written as python notebooks and located in the
*example* folder. They should be verified with the [py.test](https://pypi.python.org/pypi/pytest-ipynb) command.

At any time *ALL* tests in the *tests* folder should pass, meaning that
```bash
python -m unittest discover tests/
```
should give no errors.
