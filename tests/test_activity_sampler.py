
import unittest
import os
try:
    from cemc.mcmc import ActivitySampler
    from cemc.mcmc.mpi_tools import mpi_communicator
    from cemc import CE
    from ase.clease import CEBulk
    from ase.clease import Concentration
    from ase.clease import CorrFunction
    available = True
    reason = ""
    comm = mpi_communicator()
except Exception as exc:
    reason = str(exc)
    print(reason)
    available = False
    comm = None

class TestActivitySampler(unittest.TestCase):
    def test_no_throw(self):
        no_throw = True
        if not available:
            self.skipTest(reason)
            return
        msg = ""
        try:
            conc = Concentration(basis_elements=[["Al", "Mg"]])
            kwargs = {
                "crystalstructure": "fcc",
                "a": 4.05, "size": [4, 4, 4],
                "concentration": conc,
                "db_name": "data/temporary_bcnucleationdb_activity.db",
                "max_cluster_size": 3,
                "max_cluster_dia": 4.5
            }
            ceBulk = CEBulk(**kwargs)
            ceBulk.reconfigure_settings()
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)
            ecis = {key: 1.0 for key in cf.keys()}
            atoms = ceBulk.atoms.copy()
            calc = CE(atoms, ceBulk, ecis)

            T = 500
            c_mg = 0.4
            comp = {"Mg": c_mg, "Al": 1.0-c_mg}
            calc.set_composition(comp)
            act_sampler = ActivitySampler(atoms, T,
                                        moves=[("Al", "Mg")], mpicomm=comm)
            act_sampler.runMC(mode="fixed", steps=1000)
            act_sampler.get_thermodynamic()
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)
        os.remove("data/temporary_bcnucleationdb_activity.db")


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
