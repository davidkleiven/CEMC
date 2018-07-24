# Script that runs tests

# Python 2 tests
rm *.db *.pkl
python2 -m pip install -e ./ --user
python2 tests/runner.py

# Python 3 tests
rm *.db *.pkl
