"""
This example shows how one can use the Monte Carlo observers
"""

# First we import the BulkCrystal object from ASE
from ase.ce import BulkCrystal

# Hard-code the ECIs for simplicity
eci = {"c3_2000_5_000": -0.000554493287657111,
"c2_1000_1_00": 0.009635318249739103,
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
ceBulk = calc.BC
ceBulk.atoms.set_calculator(calc)

conc = {
    "Al":0.8,
    "Mg":0.2
}
calc.set_composition(conc)

# Now we import the Monte Carlo class
from cemc.mcmc.montecarlo import Montecarlo
T = 400 # Run the simulation at 400K
mc_obj = Montecarlo( ceBulk.atoms, T )

# Now we define the observers
from cemc.mcmc import CorrelationFunctionTracker, PairCorrelationObserver, Snapshot, LowestEnergyStructure, NetworkObserver

# The Correlation Function Tracker computes the thermodynamic
# average of all the correlation functions
corr_func_obs = CorrelationFunctionTracker(calc)

# The PairCorrelationObserver computes the thermodynamic average of all
# the pair interactions.
# If only the pair interactions are of interest, this observer should be
# preferred over the CorrelationFunctionTracker
pair_obs = PairCorrelationObserver(calc)

# This function takes a snap shot of the system and collects
# them in a trajectory file
snapshot = Snapshot( trajfile="demo.traj", atoms=ceBulk.atoms )

# This observer stores the structure having the lowest energy
low_en = LowestEnergyStructure( calc, mc_obj )

# This observer tracks networks of a certain atom type
network_obs = NetworkObserver( calc=calc, cluster_name="c2_1000_1_00", element="Mg" )

# Now we can attach the observers to the mc_obj
mc_obj.attach( corr_func_obs, interval=1 )
mc_obj.attach( pair_obs, interval=1 )
mc_obj.attach( snapshot, interval=10 ) # Take a snap shot every then iteration
mc_obj.attach( low_en, interval=1 )
mc_obj.attach( network_obs, interval=5 )

# Now run 30 MC steps
mc_obj.runMC( mode="fixed", steps=30, equil=False )
