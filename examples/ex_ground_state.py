
"""
This example shows how the ground state of a structure can be found
using Cluster Expansion

Features
1. Initialize a CEBulk object
2. Initialize a Cluster Expansion calculator
3. Initialize a the Monte Carlo object
4. Simple use of Monte Carlo observers
"""

# First we import the CEBulk object from ASE
from ase.clease import CEBulk
from ase.clease import Concentration
from util import get_example_ecis

# Initialize the CEBulk object for a 4x4x4 expansion of a
# primitive FCC unitcell
conc = Concentration([["Al","Mg"]])
bc = CEBulk(crystalstructure="fcc", a=4.05,
            db_name="test_gs_db.db", size=[3, 3, 3],
            concentration=conc, max_cluster_size=3,
            max_cluster_dia=4.5)
bc.reconfigure_settings()  # Nessecary for unittests to pass

# Just use some example ECIs
eci = get_example_ecis(bc=bc)

# Initialize a Cluster Expansion calculator (C++ version is required)
from cemc import CE
atoms = bc.atoms.copy()
calc = CE(atoms, bc, eci)

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
    mc_obj = Montecarlo(atoms, T)

    # Give the Lowest Energy Structure a reference to the monte carlo object
    obs.mc_obj = mc_obj

    # Attach the observer to the monte carlo object
    mc_obj.attach( obs )

# The Ground State Atoms object can now be obtained
gs_atoms = obs.lowest_energy_atoms


# A simpler approach that essentially follow exactly the same procedure
# as above is to use the GSFinder class
from cemc.tools import GSFinder
gs_search = GSFinder()
gs = gs_search.get_gs(bc, eci, temps=temps, n_steps_per_temp=100, composition=composition)

"""
gs is now a dictionary with the following form
{
    "atoms": the structure having the lowest energy
    "energy": Energy of that structure
    "cf": The correlation functions of the structure with the lowest energy
}
"""
