from cemc.mcmc import Montecarlo
from cemc.mcmc import LowestEnergyStructure
from cemc.wanglandau.ce_calculator import CE
import numpy as np

class GSFinder(object):
    def __init__( self ):
        pass

    def get_gs( self, BC, ecis, composition=None, temps=None, n_steps_per_temp=1000 ):
        """
        Computes the ground states

        :param BC: Instance of *BulkCrystal* or *BulkSpacegroup* from ASE
        :param ecis: Dictionary with the Effecitve Cluster Interactions
        :param composition: Dictionary with compositions (i.e. {"Mg":0.2,"Al":0.8})
        :param temps: List of cooling temperatures
        :param n_steps_per_temp: Number of MC steps per temperature
        """
        calc = CE( BC, ecis )
        BC.atoms.set_calculator( calc )
        #print (calc.get_cf())
        if ( temps is None ):
            temps = np.linspace( 1, 1500, 30 )[::-1]
        if ( composition is not None ):
            calc.set_composition( composition )

        minimum_energy = LowestEnergyStructure( calc, None )
        for T in temps:
            print ("Temperature {}".format(T) )
            mc_obj = Montecarlo( BC.atoms, T )
            minimum_energy.mc_obj = mc_obj
            mc_obj.attach( minimum_energy )
            mc_obj.runMC( steps=n_steps_per_temp, verbose=False, equil=False )
            thermo = mc_obj.get_thermodynamic()

        result = {
            "atoms":minimum_energy.atoms,
            "energy":minimum_energy.lowest_energy,
            "cf":minimum_energy.lowest_energy_cf
        }
        return result
