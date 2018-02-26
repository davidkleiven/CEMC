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
