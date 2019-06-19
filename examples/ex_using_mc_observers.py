"""
This example shows how one can use the Monte Carlo observers
"""

# First we import the CEBulk object from ASE
from ase.clease import CEBulk
from ase.clease import Concentration
from util import get_example_ecis, get_example_network_name

db_name = "database_with_dft_structures.db"

# In order to be able to construct a large Monte Carlo cell we have to
# but the arguments used to construct the CEBulk object in a
# dictionary
conc = Concentration(basis_elements=[["Al","Mg"]])
kwargs = {
    "crystalstructure":"fcc",
    "a":4.05,
    "size":[3, 3, 3],
    "db_name": db_name,
    "max_cluster_size": 3,
    "concentration": conc,
    "max_cluster_dia": 4.5
}

# In this example, we just use some example ecis
eci = get_example_ecis(bc_kwargs=kwargs)

# Initialize a template CEBulk Object
ceBulk = CEBulk( **kwargs )
ceBulk.reconfigure_settings()  # Nessecary for the unittests to pass
# Now we want to get a Cluster Expansion calculator for a big cell
mc_cell_size = [10,10,10]
from cemc import get_atoms_with_ce_calc

atoms = get_atoms_with_ce_calc(ceBulk, kwargs, eci=eci, size=mc_cell_size, db_name="mc_obs.db")
calc = atoms.get_calculator()
conc = {
    "Al":0.8,
    "Mg":0.2
}
calc.set_composition(conc)

# Now we import the Monte Carlo class
from cemc.mcmc.montecarlo import Montecarlo
T = 400 # Run the simulation at 400K
mc_obj = Montecarlo(atoms, T)

# Now we define the observers
from cemc.mcmc import CorrelationFunctionTracker, PairCorrelationObserver, Snapshot, LowestEnergyStructure, NetworkObserver
from cemc.mcmc import SiteOrderParameter, EnergyEvolution, EnergyHistogram
from cemc.mcmc import MCBackup, DiffractionObserver

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
snapshot = Snapshot( trajfile="demo.traj", atoms=atoms )

# This observer stores the structure having the lowest energy
low_en = LowestEnergyStructure( calc, mc_obj )

# This observer tracks networks of a certain atom type
network_obs = NetworkObserver( calc=calc, cluster_name=[get_example_network_name(ceBulk)], element=["Mg"])

# This tracks the average number of sites that where the symbol as changed
site_order = SiteOrderParameter(atoms)

# Energy evolution. Useful to check if the system has been equilibrated
energy_evol = EnergyEvolution(mc_obj)

# Energy histogram: Sample a histogram of the visited energy states
energy_hist = EnergyHistogram(mc_obj, n_bins=100)

# Make backup at regular intervals
mc_backup = MCBackup(mc_obj, backup_file="montecarlo_example_backup.pkl", db_name="mc_ex_backup.db")

# Diffraction observer
diffract = DiffractionObserver(atoms=atoms, k_vector=[1.0, 1.0, 1.0], 
                              active_symbols=["Mg"], all_symbols=["Al", "Mg"])

# Now we can attach the observers to the mc_obj
mc_obj.attach(corr_func_obs, interval=1)
mc_obj.attach(pair_obs, interval=1)
mc_obj.attach(snapshot, interval=10)  # Take a snap shot every then iteration
mc_obj.attach(low_en, interval=1)
mc_obj.attach(network_obs, interval=5)
mc_obj.attach(site_order)
mc_obj.attach(energy_evol)
mc_obj.attach(energy_hist)
mc_obj.attach(mc_backup, interval=5)
mc_obj.attach(diffract)

# Now run 30 MC steps
mc_obj.runMC( mode="fixed", steps=30, equil=False )

# Get statistice from the Network Observer
cluster_stat = network_obs.get_statistics()

# Compute the surface of the clusters
surf = network_obs.surface()

# Get the average number of sites changed and the standard deviation
avg_changed, std_changed = site_order.get_average()

# If we want to store an Monte carlo object and read it back later
mc_obj.save("montecarlo.pkl")

# To read it back again
mc_obj = Montecarlo.load("montecarlo.pkl")

# Remove the database
# NOTE: don't do this if you intend to rerun a similar calculation
import os
os.remove("mc_obs.db")
os.remove("montecarlo.pkl")
os.remove("montecarlo_example_backup.pkl")
os.remove("mc_ex_backup.db")
