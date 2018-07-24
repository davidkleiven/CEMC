import unittest
import numpy as np
try:
    from cemc.tools import CanonicalFreeEnergy
    available = True
except ImportError:
    available = False

class TestCanonicalFreeEnergy( unittest.TestCase ):
    def test_free_energy(self):
        if ( not available ):
            self.skipTest( "Test not available" )
            return
        T = np.linspace(100,1000,40)
        energy = 0.1*T**2

        msg = ""
        no_throw = True
        comp = {
            "Al":0.5,
            "Mg":0.5
        }
        try:
            free_eng = CanonicalFreeEnergy( comp )
            temp, U, F = free_eng.get( T, energy )
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
