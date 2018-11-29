import unittest
import os
from ase.build import bulk
try:
    from ase.clease import CEBulk
    from ase.clease import Concentration
    from ase.clease import CorrFunction
    from cemc.wanglandau import WangLandauInit, WangLandau, WangLandauDBManager
    from cemc.wanglandau import AtomExistsError
    has_CE = True
except Exception as exc:
    print (exc)
    has_CE = False

    # Define a dummy class in case of import error
    class Concentration(object):
        def __init__(self, basis_elements=None):
            pass

db_name = "temp_db_wanglandau.db"
wl_db_name = "wanglandau_test_init.db"
conc = Concentration(basis_elements=[["Al", "Mg"]])
bc_kwargs = {
    "crystalstructure": "fcc",
    "size": [3, 3, 3],
    "concentration": conc,
    "db_name": db_name,
    "max_cluster_size": 3,
    "a": 4.05
}


def get_eci():
    bc = CEBulk(**bc_kwargs)
    bc.reconfigure_settings()
    cf = CorrFunction(bc)
    cf = cf.get_cf(bc.atoms)
    eci = {key: 0.001 for key in cf.keys()}
    return eci

class TestInitWLSim(unittest.TestCase):
    def test_no_throw(self):
        no_throw = True
        if not has_CE:
            self.skipTest("ASE version does not have CE")

        try:
            os.remove("temp_db.db")
        except Exception:
            pass
        msg = ""
        eci = get_eci()
        try:
            initializer = WangLandauInit(wl_db_name)
            T = [1000, 10]
            comp = {"Al": 0.5, "Mg": 0.5}
            try:
                initializer.insert_atoms(
                    bc_kwargs, size=[5, 5, 5],
                    T=T, n_steps_per_temp=10, eci=eci, composition=comp)
            except AtomExistsError:
                pass
            initializer.prepare_wang_landau_run( [("id","=","1")] )
            atoms = initializer.get_atoms(1, eci)
            db_manager = WangLandauDBManager(wl_db_name)
            runID = db_manager.get_next_non_converged_uid( 1 )
            if ( runID == -1 ):
                raise ValueError( "No new Wang Landau simulation in the database!" )
            simulator = WangLandau(atoms, wl_db_name, runID, fmin=1.8)
            simulator.run_fast_sampler(mode="adaptive_windows", maxsteps=100)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
