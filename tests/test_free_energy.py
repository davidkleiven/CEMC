import unittest
import json
from cemc.tools import free_energy as fe

with open( "tests/test_data/free_energy_test_data.json", 'r') as infile:
    data = json.load( infile )

internal_energy = data["energy"]
singlets = {"c1_1":data["singlets"]} # c1_1 is just the name of the cluster
T = data["temperature"]
mu = {"c1_1":data["mu"]}

class TestFreeEnergy( unittest.TestCase ):
    def test_no_throw( self ):
        no_throw = True
        try:
            free = fe.FreeEnergy()
            N_atoms = 1000.0
            sgc_energy = free.get_sgc_energy( internal_energy, singlets, mu )
            G = free.free_energy_isochemical( T=T, sgc_energy=sgc_energy/N_atoms, nelem=2 )
            H = free.helmholtz_free_energy( G["free_energy"], mu, singlets )
        except Exception as exc:
            print (str(exc))
            no_throw = False
        self.assertTrue( no_throw )

if __name__ == "__main__":
    unittest.main()
