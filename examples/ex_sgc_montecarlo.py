"""
This example shows how one can run a Monte Carlo simulation in the
Semi-Grand-Canonical ensemble and some basic postprocessing

Features
1. Initialize a BulkCrystal objects from a dictionary of arguments
2. Get a large cell for Monte Carlo and a corresponding Cluster Expansion
   calculator
3. Extract thermodynamic data from a Monte Carlo simulation
"""

# First we import the BulkCrystal object from ASE
from ase.ce import BulkCrystal
from util import get_example_ecis

# Specify the concentration arguments (they really don't matter here)
# They only have effect when generating structure of Cluster Expansion
conc_args = {
            "conc_ratio_min_1":[[1,0]],
            "conc_ratio_max_1":[[0,1]],
        }

db_name = "database_with_dft_structures.db"

# In order to be able to construct a large Monte Carlo cell we have to
# but the arguments used to construct the BulkCrystal object in a
# dictionary
kwargs = {
    "crystalstructure":"fcc",
    "a": 4.05,
    "size": [3, 3, 3],
    "basis_elements": [["Al","Mg"]],
    "db_name": db_name,
    "conc_args": conc_args,
    "max_cluster_size": 3
}

# Use some example ecis
eci = get_example_ecis(bc_kwargs=kwargs)

# Initialize a template BulkCrystal Object
ceBulk = BulkCrystal( **kwargs )
ceBulk.reconfigure_settings()  # Nessecary for the unittest to pass

# Now we want to get a Cluster Expansion calculator for a big cell
mc_cell_size = [10, 10, 10]
from cemc import get_ce_calc

calc = get_ce_calc( ceBulk, kwargs, eci=eci, size=mc_cell_size )

# Now er are finished with the template BulkCrystal
ceBulk = calc.BC
ceBulk.atoms.set_calculator( calc )

# In the SGC ensemble the simulation is run at fixed chemical potential
# The chemical potentials are subtracted from the singlet terms
# Those are ECIs starting with c1.
# In a binary system there is only one singlet term, so there is only one
# chemical potential to specify
chem_pot = {
    "c1_0":-1.04
}

# Speciy the temperature
T = 400

from cemc.mcmc import SGCMonteCarlo

mc = SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg"] )

# To make the exampl fast, we don't equillibriate the system
# In general, it is a good idea to equillibriate the system
equillibriate = False
mc.runMC( steps=100, chem_potential=chem_pot, equil=equillibriate)

# To extract the thermo dynamic properties from the simulations
thermo_prop = mc.get_thermodynamic()
