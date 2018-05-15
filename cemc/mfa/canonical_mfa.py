from ase.units import kB
import numpy as np

class CanonicalMeanField(object):
    def __init__( self, atoms=None, T=[100] ):
        self.atoms = atoms
        self.calc = self.atoms._calc # Should be a cluster expansion calculator
        self.T = np.array(T).astype(np.float64)
        self.E0 = self.calc.get_energy()
        self.mean_energy = np.zeros_like(self.T)
        self.energy_squared = np.zeros_like(self.T)
        self.Z = np.zeros_like(self.T)
        self.warn_new_ground_state = True

        # Store the atoms and the correlation functions of a new ground state
        # Ideally these variables should never be set
        # If they are set one can run a new mean field calculation around these
        self.gs_atoms = None
        self.gs_cf = None
        self.gs_energy = 0.0

    def contrib_one_atom( self, ref_indx ):
        """
        Calculate the contribution to the total energy
        """
        Z_single = np.zeros_like(self.T)
        ref_symb = self.atoms[ref_indx].symbol
        E_single = np.zeros_like(self.T)
        E_squared_single = np.zeros_like(self.T)
        beta = 1.0/(kB*self.T)

        for i in range(len(self.atoms)):
            if ( i == ref_indx ):
                continue

            if ( self.atoms[i].symbol == ref_symb ):
                Z_single += 1.0
                new_E = self.calc.get_energy()-self.E0
                E_single += new_E
                E_squared_single += new_E**2
            else:
                new_symb = self.atoms[i].symbol
                system_changes = [(ref_indx,ref_symb,new_symb),(i,new_symb,ref_symb)]
                new_E = self.calc.calculate( self.atoms, ["energy"], system_changes )-self.E0
                if ( new_E < self.gs_energy ):
                    self.warning_new_ground_state( new_E )
                    self.gs_atoms = self.atoms.copy()
                    self.gs_cf = self.calc.get_cf()
                    self.gs_energy = new_E
                boltzmann = np.exp(-beta*new_E)
                E_single += new_E*boltzmann
                E_squared_single += new_E**2*boltzmann
                Z_single += boltzmann
                self.calc.undo_changes()
                self.calc.clear_history()
                self.atoms[ref_indx].symbol = ref_symb
                self.atoms[i].symbol = new_symb
        return Z_single,E_single,E_squared_single,

    def warning_new_ground_state(self, dE ):
        """
        Prints a warning message that a state with lowest energy was found
        """
        if ( self.warn_new_ground_state ):
            print ( "A structure with with energy: {} eV lower than the current ground state was found".format(dE) )
            print ( "Normally one should run the Mean Field Approximation as a perturbation around the ground state" )
            self.warn_new_ground_state = False

    def reset(self):
        self.Z = np.zeros_like(self.T)
        self.mean_energy = np.zeros_like(self.T)
        self.energy_squared = np.zeros_like(self.T)

    def calculate( self ):
        self.reset()
        for i in range(len(self.atoms)):
            Z_single,E_single,E_squared_single = self.contrib_one_atom(i)
            self.Z += Z_single
            self.mean_energy += E_single
            self.energy_squared += E_squared_single
        self.mean_energy /= self.Z
        self.energy_squared /= self.Z

        result = {}
        result["partition_function"] = self.Z
        result["internal_energy"] = self.E0 + self.mean_energy
        result["free_energy"] = self.E0 - kB*self.T*np.log(self.Z)
        return result
