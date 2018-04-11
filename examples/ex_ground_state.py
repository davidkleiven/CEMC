
"""
This example shows how the ground state of a structure can be found
using Cluster Expansion

Features
1. Initialize a BulkCrystal object
2. Initialize a Cluster Expansion calculator
3. Initialize a the Monte Carlo object
4. Simple use of Monte Carlo observers
"""

# First we import the BulkCrystal object from ASE
from ase.ce import BulkCrystal

# Initialize the BulkCrystal object for a 4x4x4 expansion of a
# primitive FCC unitcell
conc_args = {
    "conc_ratio_min_1":[[0,1]],
    "conc_ratio_max_1":[[1,0]]
}
bc = BulkCrystal( crystalstructure="fcc", a=4.05, conc_args=conc_args, db_name="test_gs_db.db", size=[4,4,4], basis_elements=[["Al","Mg"]] )

# Get the Effective Cluster Interactions (here they are hard coded),
# but these are typically read from a file
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

# Initialize a Cluster Expansion calculator (C++ version is required)
from cemc.wanglandau import CE
calc = CE( bc, eci )
bc.atoms.set_calculator(calc)

# NOTE: At this point all changes to the atoms object has to be done via
# the calculator. The reason is that the calculator keeps track of the
# changes in the atoms object.

mg_conc = 0.25 # Magnesium concentration

composition = {
    "Mg":mg_conc,
    "Al":1.0-mg_conc
}

# Set the composition
calc.set_composition(composition)

# Define temperatures (in a real applciation consider more temperatures)
temps = [800,700,600,500,400,300,200,100]

# Define the number of steps per temperature
n_steps_per_temp = 100 # In a real application condier more that this

# We have to keep track of the lowest structure. This can be done
# adding an observer to the Monte Carlo that always store the
# lowest energy structure
from cemc.mcmc.mc_observers import LowestEnergyStructure
obs = LowestEnergyStructure( calc, None )

# Now we import the Monte Carlo class
from cemc.mcmc.montecarlo import Montecarlo

# Loop over temperatures
for T in temps:
    mc_obj = Montecarlo( bc.atoms, T )

    # Give the Lowest Energy Structure a reference to the monte carlo object
    obs.mc_obj = mc_obj

    # Attach the observer to the monte carlo object
    mc_obj.attach( obs )

# The Ground State Atoms object can now be obtained
gs_atoms = obs.lowest_energy_atoms
