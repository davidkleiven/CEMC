from ase.calculators.calculator import Calculator
from ase.ce.corrFunc import CorrFunction
from ase.ce.settings import BulkCrystal

class CEcalc( Calculator ):
    implemented_properties = ['energy']

    def __init__( self, ecis, BC ):
        Calculator.__init__(self)
        self.corrFunc = CorrFunction( BC )
        self.eci = ecis

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms)

        cf_by_name = self.corrFunc.get_cf_by_cluster_names( atoms, self.eci.keys() )
        energy = 0.0
        for key,value in self.eci.iteritems():
            energy += value*cf_by_name[key]
        self.results["energy"] = energy*len(atoms) # Should not return energy per atom!
