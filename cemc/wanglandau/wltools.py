import io
import numpy as np
import sqlite3 as sq
from ase.data import chemical_symbols

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out,arr)
    out.seek(0)
    return sq.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def key_value_lists_to_dict( keys, values ):
    dictionary = {}
    for i in range(len(keys)):
        dictionary[keys[i]] = values[i]
    return dictionary

def element_count( atoms ):
    """
    Counts the number of each element in the atoms object
    """
    res = {}
    for atom in atoms:
        if ( not atom.symbol in res.keys() ):
            res[atom.symbol] = 1
        else:
            res[atom.symbol] += 1
    return res

def get_formula( atomic_numbers ):
    elm_count = {}
    for num in atomic_numbers:
        if ( num in elm_count.keys() ):
            elm_count[num] += 1
        else:
            elm_count[num] = 1

    formula = ""
    for key,value in elm_count.items():
        formula += "{}{}".format(chemical_symbols[key],value)
    return formula
