import montecarlo as mc
from cemc.mcmc.mc_observers import SGCObserver
import numpy as np
from ase.units import kB
import copy

class SGCMonteCarlo( mc.Montecarlo ):
    def __init__( self, atoms, temp, indeces=None, symbols=None, mpicomm=None ):
        mc.Montecarlo.__init__( self, atoms, temp, indeces=indeces, mpicomm=mpicomm )
        if ( not symbols is None ):
            # Override the symbols function in the main class
            self.symbols = symbols
        self.averager = SGCObserver( self.atoms._calc, self, len(self.symbols)-1 )
        self.chem_pots = []
        self.chem_pot_names = []
        self.has_attached_avg = False

    def get_trial_move( self ):
        indx = np.random.randint( low=0, high=len(self.atoms) )
        old_symb = self.atoms[indx].symbol
        new_symb = old_symb
        while( new_symb == old_symb ):
            new_symb = self.symbols[np.random.randint(low=0,high=len(self.symbols))]
        system_changes = [(indx,old_symb,new_symb)]
        return system_changes

    def update_tracker( self, system_changes ):
        """
        Override the update of the atom tracker. The atom tracker is irrelevant in the semi grand canonical ensemble
        """
        pass

    @property
    def chemical_potential( self ):
        return self.__chemical_potential

    @chemical_potential.setter
    def chemical_potential( self, chem_pot ):
        self.__chemical_potential = chem_pot
        self.reset_eci_to_original( self.atoms._calc.eci)
        self.include_chemcical_potential_in_ecis( chem_pot, self.atoms._calc.eci )

    def include_chemcical_potential_in_ecis( self, chem_potential, eci ):
        """
        Including the chemical potentials in the ecis
        """
        self.chem_pots = []
        self.chem_pot_names = []
        keys = chem_potential.keys()
        keys.sort()
        for key in keys:
            self.chem_pots.append( chem_potential[key] )
            self.chem_pot_names.append(key)
            eci[key] -= chem_potential[key]
        print (eci)
        self.atoms._calc.update_ecis( eci )
        return eci

    def reset_eci_to_original( self, eci_with_chem_pot ):
        """
        Resets the ecis to their original value
        """
        for name,val in zip(self.chem_pot_names,self.chem_pots):
            eci_with_chem_pot[name] += val
        self.atoms._calc.update_ecis( eci_with_chem_pot )
        return eci_with_chem_pot

    def runMC( self, steps = 10, verbose = False, chem_potential=None, equil=True ):
        if ( chem_potential is None and self.chemical_potential is None ):
            ex_chem_pot = {
                "c1_1":-0.1,
                "c1_2":0.05
            }
            raise ValueError( "No chemicla potentials given. Has to be dictionary of the form {}".format(ex_chem_pot) )
        if ( chem_potential is not None ):
            self.chemical_potential = chem_potential
        self.averager.reset()

        if ( equil ):
            self.equillibriate()
            self.averager.reset()

        if ( not self.has_attached_avg ):
            self.attach( self.averager )
            self.has_attached_avg = True
        mc.Montecarlo.runMC( self, steps=steps, verbose=verbose, equil=False )

        # Reset the chemical potential to zero
        zero_chemical_potential = {key:0.0 for key in self.chemical_potential.keys()}
        self.chemical_potential = zero_chemical_potential
        #eci = self.reset_eci_to_original( eci )
        #self.atoms._calc.update_ecis( eci )

    def collect_averager_results(self):
        """
        If MPI is used, this function collects the results from the averager
        """
        if ( self.mpicomm is None ):
            return

        size = self.mpicomm.Get_size()
        all_res = self.mpicomm.gather( self.averager.quantities, root=0 )
        rank = self.mpicomm.Get_rank()
        if ( rank == 0 ):
            self.averager.quantities = all_res[0]
            for i in range(1,len(all_res)):
                for key,value in all_res[i].iteritems():
                    self.averager.quantities[key] += value

                # Normalize by the number of processors
                for key in self.averager.quantities.keys():
                    self.averager[key] /= size

    def get_thermodynamic( self ):
        N = self.averager.counter
        quantities = {}
        quantities["singlets"] = self.averager.singlets/N
        quantities["chem_pots"] = self.chem_pots
        quantities["energy"] = self.averager.energy/N
        for i in range( len(quantities["chem_pots"]) ):
            quantities["energy"] += quantities["chem_pots"][i]*quantities["singlets"][i]

        quantities["heat_capacity"] = self.averager.energy_sq/N - (self.averager.energy/N)**2 + \
                                      np.sum( self.averager.singl_eng/N - (self.averager.energy/N)*quantities["singlets"] )
        quantities["heat_capacity"] /= (kB*self.T**2)
        quantities["temperature"] = self.T
        return quantities
