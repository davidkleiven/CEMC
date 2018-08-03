"""
This example shows how one can use the Monte Carlo observers
"""

# First we import the BulkCrystal object from ASE
from ase.ce import BulkCrystal
from util import get_example_ecis, get_example_network_name

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
    "size":[3, 3, 3],
    "basis_elements":[["Al","Mg"]],
    "db_name": db_name,
    "conc_args": conc_args,
    "max_cluster_size": 3
}

# In this example, we just use some example ecis
eci = get_example_ecis(bc_kwargs=kwargs)

# Initialize a template BulkCrystal Object
ceBulk = BulkCrystal( **kwargs )
ceBulk.reconfigure_settings()  # Nessecary for the unittests to pass
# Now we want to get a Cluster Expansion calculator for a big cell
mc_cell_size = [10,10,10]
from cemc import get_ce_calc

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
from cemc.mcmc import SiteOrderParameter

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
network_obs = NetworkObserver( calc=calc, cluster_name=get_example_network_name(ceBulk), element="Mg" )

# This tracks the average number of sites that where the symbol as changed
site_order = SiteOrderParameter(ceBulk.atoms)

# Now we can attach the observers to the mc_obj
mc_obj.attach( corr_func_obs, interval=1 )
mc_obj.attach( pair_obs, interval=1 )
mc_obj.attach( snapshot, interval=10 ) # Take a snap shot every then iteration
mc_obj.attach( low_en, interval=1 )
mc_obj.attach( network_obs, interval=5 )
mc_obj.attach(site_order)

# Now run 30 MC steps
mc_obj.runMC( mode="fixed", steps=30, equil=False )

# Get statistice from the Network Observer
cluster_stat = network_obs.get_statistics()

# Compute the surface of the clusters
surf = network_obs.surface()

# Get the average number of sites changed and the standard deviation
avg_changed, std_changed = site_order.get_average()
