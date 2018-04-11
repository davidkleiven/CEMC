"""
This example shows how one can compute the Free Energy along a line
of constant chemical potential
"""

# JSON file containing example data
fname = "examples/example_sgcmc_results.json"

import json
with open(fname,'r') as infile:
    data = json.load(infile)

# The only dataset in the file was named mu10720
# The number atoms was 1000
data = data["mu10720m"]
n_atoms = 1000

from cemc.tools import FreeEnergy
import numpy as np

# We have to put the singlets and chemical potential in a dictionary
# This is useful when there are multiple chemical potentials, but may
# seem a bit complicated when there is only one
# NOTE: The two dictonary should have exactly the same keys
singlets = {
    "c1":np.array(data["singlets"])
}
mu = {
    "c1":data["mu"]
}

internal_energy = np.array( data["energy"] )

free_eng = FreeEnergy()

# Now we can get the free energy along the line of constant chemical potential
# The number of elements in this simulation is 2 (it was only Al and Mg)
res = free_eng.free_energy_isochemical( T=data["temperature"], sgc_energy=internal_energy/n_atoms, nelem=2 )

# We could also compute the Helmholtz Free Energy along a line of constant
# chemical potential. Remember that the composition changes along
# this line
# To do so we have to sort the singlet terms
singlets["c1"] = np.array( [singlets["c1"][indx] for indx in res["order"]])
F = free_eng.helmholtz_free_energy( res["free_energy"], singlets, mu )

# Plot the results
from matplotlib import pyplot as plt
fig, ax = plt.subplots(ncols=2)
ax[0].plot( res["temperature"], res["free_energy"], marker="o", label="SGC" )
ax[0].plot( res["temperature"], F, marker="x", label="Helmholtz")
ax[0].legend()
ax[1].plot( res["temperature"], singlets["c1"] )

ax[0].set_xlabel( "Temperature (K)" )
ax[1].set_xlabel( "Temperature (K)" )
ax[0].set_ylabel( "Free energy (eV/atom)")
ax[1].set_ylabel( "Singlets" )

if __name__ == "__main__":
    plt.show()
