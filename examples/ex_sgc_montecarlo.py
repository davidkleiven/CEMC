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

# Hard-code the ECIs for simplicity
eci = {"c3_2000_5_000": -0.000554493287657111,
"c2_1000_1_00": 0.009635318249739103,
"c3_2000_3_000": -0.0012517824048219194,
"c3_1732_1_000": -0.0012946400900521093,
"c2_1414_1_00": -0.017537890489630819,
"c4_1000_1_0000": -1.1303654231631574e-05,
"c3_2000_4_000": -0.00065595035208737659,
"c2_1732_1_00": -0.0062866523139031511,
"c4_2000_11_0000": 0.00073748615657533178,
"c1_0": -1.0685540954294481,
"c4_1732_8_0000": 6.2192225273001889e-05,
"c3_1732_4_000": -0.00021105632231802613,
"c2_2000_1_00": -0.0058771555942559303,
"c4_2000_12_0000": 0.00026998290577185763,
"c0": -2.6460470182744342,
"c4_2000_14_0000": 0.00063004101881374334,
"c4_1414_1_0000": 0.00034847251116721441}

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
    "a":4.05,
    "size":[4,4,4],
    "basis_elements":[["Al","Mg"]],
    "db_name":db_name,
    "conc_args":conc_args
}

# Initialize a template BulkCrystal Object
ceBulk = BulkCrystal( **kwargs )

# Now we want to get a Cluster Expansion calculator for a big cell
mc_cell_size = [10,10,10]
from cemc.wanglandau.ce_calculator import get_ce_calc

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
