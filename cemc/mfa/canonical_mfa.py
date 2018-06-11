from ase.units import kB
import numpy as np
import time

class CanonicalMeanField(object):
    def __init__( self, atoms=None, T=[100] ):
        self.atoms = atoms
        self.calc = self.atoms._calc # Should be a cluster expansion calculator
        self.T = np.array(T).astype(np.float64)
        self.E0 = self.calc.get_energy()
        self.mean_energy = np.zeros_like(self.T)
        self.energy_squared = np.zeros_like(self.T)
        self.Z = np.ones_like(self.T)
        self.warn_new_ground_state = True
        self.status_every = 30

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

        for i in range(ref_indx+1,len(self.atoms)):
            if ( self.atoms[i].symbol == ref_symb ):
                # Do nothing partition function is the sum over all different states
                pass
                #Z_single += 1.0
                #new_E = self.calc.get_energy()-self.E0
                #E_single += new_E
                #E_squared_single += new_E**2
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
        return Z_single,E_single,E_squared_single

    def relax(self):
        """
        Relax by single atoms swaps
        """
        system_changed = True
        iteration = 0
        now = time.time()
        while ( system_changed ):
            self.log( "Relaxing system: iteration {}".format(iteration) )
            iteration += 1
            system_changed = False
            for ref_indx in range(len(self.atoms)):
                ref_symb = self.atoms[ref_indx].symbol
                if ( time.time()-now > self.status_every ):
                    self.log( "Currently at ref atom {} of {}".format(ref_indx,len(self.atoms)) )
                    now = time.time()
                for i in range(len(self.atoms)):
                    if ( self.atoms[ref_indx].symbol == self.atoms[i].symbol ):
                        continue
                    new_symb = self.atoms[i].symbol
                    system_changes = [(ref_indx,ref_symb,new_symb),(i,new_symb,ref_symb)]
                    energy = self.calc.calculate( self.atoms, ["energy"], system_changes )
                    if ( energy < self.E0 ):
                        self.E0 = energy
                        system_changed = True
                        self.log ("New ground state energy: {} eV".format(self.E0))
                        break
                    else:
                        self.calc.undo_changes()
                        self.calc.clear_history()
                        self.atoms[ref_indx].symbol = ref_symb
                        self.atoms[i].symbol = new_symb
                if ( system_changed ):
                    break

    def warning_new_ground_state(self, dE ):
        """
        Prints a warning message that a state with lowest energy was found
        """
        if ( self.warn_new_ground_state ):
            print ( "A structure with with energy: {} eV lower than the current ground state was found".format(dE) )
            print ( "Normally one should run the Mean Field Approximation as a perturbation around the ground state" )
            self.warn_new_ground_state = False

    def reset(self):
        self.Z = np.ones_like(self.T)
        self.mean_energy = np.zeros_like(self.T)
        self.energy_squared = np.zeros_like(self.T)

    def log( self, msg ):
        """
        Logg messages
        """
        print (msg)

    def get_concentration(self):
        """
        Compute the concentration
        """
        atoms_count = {}
        for atom in self.atoms:
            if ( atom.symbol in atoms_count.keys() ):
                atoms_count[atom.symbol] += 1
            else:
                atoms_count[atom.symbol] = 1
        conc = {key:float(value)/len(self.atoms) for key,value in atoms_count.iteritems()}
        return conc

    def calculate( self ):
        self.reset()
        self.log( "Computing partition function in the Mean Field Approximation" )
        now = time.time()
        for i in range(len(self.atoms)):
            if ( time.time() - now > self.status_every ):
                self.log("Running atom {} of {}".format(i,len(self.atoms)))
                now = time.time()
            Z_single,E_single,E_squared_single = self.contrib_one_atom(i)
            self.Z += Z_single
            self.mean_energy += E_single
            self.energy_squared += E_squared_single
        self.mean_energy /= self.Z
        self.energy_squared /= self.Z

        result = {}
        result["partition_function"] = self.Z.tolist()
        result["internal_energy"] = (self.E0 + self.mean_energy).tolist()
        result["free_energy"] = (self.E0 - kB*self.T*np.log(self.Z)).tolist()
        result["natoms"] = len(self.atoms)
        result["conc"] = self.get_concentration()
        result["temperature"] = self.T.tolist()
        return result
