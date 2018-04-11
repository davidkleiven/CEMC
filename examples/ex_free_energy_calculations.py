"""
This example shows how some basic post processing of Monte Carlo data

Features
1. Compute Helmholtz Free Energy from a MC simulation
2. Compute entropy
3. Compute enthalpy of formation
"""

# This file contains the data from a fictitous MC simulation
# The file has two columns separated by comma
# The first is tempeature the second is internal energy
fname = "examples/free_energy_data.csv"
n_atoms = 1000 # The MC simulation of the data in the file used 1000 atoms

# The chemical formula of the system in the file was Al750Mg250
conc = {
    "Mg":0.25,
    "Al":0.75
}

# Import the Canonical Free Energy class
from cemc.tools import CanonicalFreeEnergy

free_eng = CanonicalFreeEnergy(conc)

import numpy as np
temperature, internal_energy = np.loadtxt( fname, delimiter=",", unpack=True )

# The function also return internal energy and temperature
# Because the data in a file might not be sorted
# In the case of unsorted data the function will sort the data
temperature,internal_energy,helmholtz_free_energy = free_eng.get( temperature, internal_energy/n_atoms )

entropy = (internal_energy-helmholtz_free_energy)/temperature
enthalpy = helmholtz_free_energy + temperature*entropy

# Plot the results
from matplotlib import pyplot as plt
fig, ax = plt.subplots( ncols=2, nrows=2 )
ax[0,0].plot( temperature, internal_energy, marker="o" )
ax[0,1].plot( temperature, helmholtz_free_energy, marker="o" )
ax[1,0].plot( temperature, entropy, marker="o" )
ax[1,1].plot( temperature, enthalpy, marker="o")

ax[0,0].set_xlabel( "Temperature (K)" )
ax[0,1].set_xlabel( "Temperature (K)" )
ax[1,0].set_xlabel( "Temperature (K)" )
ax[1,1].set_xlabel( "Temperature (K)")
ax[0,0].set_ylabel( "Internal energy (eV/atom)" )
ax[0,1].set_ylabel( "Helmholtz Free Energy (eV/atom)" )
ax[1,0].set_ylabel( "Entropy (eV/K atom)" )
ax[1,1].set_ylabel( "Enthalpy (eV/atom)" )

if __name__ == "__main__":
    plt.show()
