from cemc.mcmc import BiasPotential
from cemc.mcmc import InertiaTensorObserver
from cemc.tools import StrainEnergy
from cemc.tools import rotate_tensor, rotate_rank4_mandel
import numpy as np

class Strain(BiasPotential):
    def __init__(self, mc_sampler=None, cluster_elements=[], C_matrix=None,
                 C_prec=None, misfit=None):
        from cemc.mcmc import FixedNucleusMC
        if not isinstance(mc_sampler, FixedNucleusMC):
            raise TypeError("mc_sampler has to be of type FixedNuceus sampler!")
        self.mc = mc_sampler
        self._volume = 10.0
        self.inert_obs = InertiaTensorObserver(atoms=self.mc.atoms, 
                                               cluster_elements=cluster_elements)
        self.mc.attach(self.inert_obs)
        self.misfit = misfit
        self.C_matrix = C_matrix
        self.C_prec = C_prec

    def initialize(self):
        """Initialize the bias potential."""
        num_solutes = self.mc.num_atoms_in_cluster
        vol_per_atom = self.mc.atoms.volume()/len(self.mc.atoms)
        self._volume = num_solutes*vol_per_atom

    def __call__(self, system_changes):
        self.inert_obs(system_changes)

        principal, rot_matrix = np.linalg.eig(self.inert_obs.inertia)
        C_mat = rotate_rank4_mandel(self.C_matrix, rot_matrix)
        C_prec = rotate_rank4_mandel(self.C_prec, rot_matrix)
        misfit = rotate_tensor(self.misfit, rot_matrix)

        str_eng = StrainEnergy(aspect=self.ellipsoid_axes(principal), 
                               misfit=misfit)
        str_energy = str_eng.strain_energy(C_matrix=C_mat, C_prec=C_prec)

        # Bias potential should not alter the observer
        # The MC object will handle this
        self.inert_obs.undo_last()
        return str_energy*self._volume

    def calculate_from_scratch(self, atoms):
        """Calculate the strain energy from scratch."""
        self.inert_obs.set_atoms(atoms)
        return self([])

    def ellipsoid_axes(self, princ):
        """Calculate the ellipsoid principal axes.
        
        :return: Numpy array of length 3 (a, b, c)
        :rtype: numpy.ndarray
        """
        princ_axes = np.zeros(3)
        princ_axes[0] = (princ[1] + princ[2] - princ[0])
        princ_axes[1] = (princ[0] + princ[2] - princ[1])
        princ_axes[2] = (princ[0] + princ[1] - princ[2])
        princ_axes *= 5.0/2.0
        princ_axes[princ_axes<0.0] = 0.0
        return np.sqrt(princ_axes)
