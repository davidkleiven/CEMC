"""
import montecarlo as mc
from cemc.mcmc.mc_observers import SGCObserver
from cemc.mcmc import SGCMonteCarlo
import numpy as np
from ase.units import kB
import copy
from scipy import stats
import mpi_tools
"""
from cemc.mcmc import SGCMonteCarlo
from cemc.mcmc.mc_observers import SGCObserver
import numpy as np

class SGCFreeEnergyBarrier( SGCMonteCarlo ):
	def __init__( self, atoms, T, **kwargs):
		n_windows = 5
		n_bins = 10
		min_singlet = 0.5
		max_singlet = 1
		if ( "n_windows" in kwargs.keys() ):
			n_windows = kwargs.pop("n_windows")
		if ( "n_bins" in kwargs.keys() ):
			n_bins = kwargs.pop("n_bins")
		if ( "min_singlet" in kwargs.keys() ):
			min_singlet = kwargs.pop("min_singlet")
		if ( "max_singlet" in kwargs.keys() ):
			max_singlet = kwargs.pop("max_singlet")
		self.n_windows = n_windows
		self.n_bins = n_bins
		self.min_singlet = min_singlet
		self.max_singlet = max_singlet

		super( SGCFreeEnergyBarrier, self ).__init__( atoms, T, **kwargs)	
		
		# Attach averager to get singlets and composition
		self.averager = SGCObserver( self.atoms._calc, self, len(self.symbols)-1 )
		self.attach( self.averager )
		
		# Set up whats needed 		
		self.window_singletrange = (self.max_singlet - self.min_singlet)/self.n_windows
		self.bin_singletrange = self.window_singletrange / self.n_bins
		# Set up data storage, all windows must have one bin extra, except the first one
		#self.data = np.zeros(self.n_windows * (self.n_bins + 1) - 1)
		self.data = [0] * (self.n_windows * (self.n_bins + 1) - 1)		
		self.current_window = 0

	def accept( self, system_changes ):
		# Check if move accepted by SGCMonteCarlo
		move_accepted = SGCMonteCarlo.accept(self, system_changes)
		# Now check if the move keeps us in same window
		#singlet = self.averager.singlets[0]
		new_singlets = np.zeros_like(self.averager.singlets)
		self.atoms._calc.get_singlets( new_singlets )
		singlet = new_singlets[0]
		# Set in_window to True and check if it should be False instead
		in_window = True
		if (self.current_window == 0):
			# Now we are in first window
			min_allowed = self.min_singlet
		else:
			#Now we are in a later window, must also allow the last bin from the previous window
			min_allowed = self.min_singlet + self.current_window * self.window_singletrange - self.bin_singletrange 
		# The maximum allowed singlet is calculated in the same way for both cases
		max_allowed = self.min_singlet + (self.current_window + 1) * self.window_singletrange		
		if (singlet < min_allowed or singlet > max_allowed):
			in_window = False
		# Now system will return to previous state if not inside window
		print (in_window)
		return move_accepted and in_window

	def run( self, nsteps = 10000 ):
		# For all windows
		for i in range(self.n_windows):
			self.current_window = i
			# We are inside a new window, update to start with concentration in the middle of this window
			newsinglet = self.min_singlet + (self.current_window + 0.5) * self.window_singletrange
			self.atoms._calc.set_singlets({"c1_0":newsinglet})
			"""
			newconc = (1 + newsinglet)/2 
			# TODO: Fix so that this is generally for atoms object and not hardcoded for Al and Mg			
			composition = {
				"Al":1.0-newconc,
				"Mg":newconc 
			}
			self.atoms._calc.set_composition(composition)			
			"""
			#Now we are in the middle of the current window, start MC
			current_step = 0
			while (current_step < nsteps):
				current_step += 1
				# Run MC step
				self._mc_step()
				# Get singlet value
				singlet = self.averager.singlets[0]
				# Subtract so that singlet value is relative to start of window
				relative_singlet = singlet - self.min_singlet - (self.current_window * self.window_singletrange) 
				# First set bin_index to beginning of the correct window
				bin_index = self.current_window * (self.n_bins) + self.current_window
 				# Now find correct bin_index from the relative singlet
				bin_index += relative_singlet // self.bin_singletrange
				# Now update the correct bin
				self.data[int(bin_index)] = self.data[int(bin_index)] + 1
				#Reset averager to be able to read singlets in accept function
				self.averager.reset()

		print(self.data)
		# Now the MC has been run for all windows and steps, must edit data for all except first window
		bin_difference = 0
		# Set range so we do not modify first window of data
		for j in range(self.n_windows - 1):
			# Calculate new difference as new window is entered
			bin_difference = self.data[self.n_bins + j * (self.n_bins + 1)] - self.data[self.n_bins + j * (self.n_bins + 1) - 1] 
			for k in range(self.n_bins + 1):
				# Remove this difference from all bins in the window
				index = self.n_bins + (j * (self.n_bins + 1)) + k
				self.data[index] = self.data[index] - bin_difference
		print(self.data)
		# Now go through data again and remove the bins that are equal
		for c in range(self.n_windows - 1):
			del self.data[(c+1) * (self.n_bins)]
		print(self.data)

	def return_umbrella_data( self ):
		# This function only returns the data
		return self.data

		
