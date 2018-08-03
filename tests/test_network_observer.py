import unittest
try:
    from cemc.mcmc import NetworkObserver
    from cemc import CE
    from ase.ce import BulkCrystal
    from ase.ce import CorrFunction
    from ase.io import read
    from ase.visualize import view
    from helper_functions import get_example_network_name
    available = True
except Exception as exc:
    print (str(exc))
    available = False

atom_files = [{
    "fname":"tests/test_data/one_root_one_clust.xyz",
    "n_roots":1,
    "size":[2]
    },
    {
    "fname":"tests/test_data/one_root_two_clust.xyz",
    "n_roots":1,
    "size":[3]
    },
    {
    "fname":"tests/test_data/three_root.xyz",
    "n_roots":3,
    "size":[]
    },
    {
    "fname":"tests/test_data/two_root_three_two_clust.xyz",
    "n_roots":2,
    "size":[3,2]
    }
    ]

class TestNetworkObs( unittest.TestCase ):
    def test_network(self):
        if ( not available ):
            self.skipTest( "ASE version does not have CE!" )
            return

        msg = ""
        no_throw = True
        try:
            db_name = "test_db_network.db"

            conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
            a = 4.05
            ceBulk = BulkCrystal(
                crystalstructure="fcc", a=a, size=[3, 3, 3],
                basis_elements=[["Al","Mg"]], conc_args=conc_args,
                db_name=db_name, max_cluster_size=3)
            ceBulk.reconfigure_settings()
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)
            eci = {key:0.001 for key in cf.keys()}
            calc = CE( ceBulk, eci=eci )
            ceBulk.atoms.set_calculator(calc)
            trans_mat = ceBulk.trans_matrix

            for name, info in ceBulk.cluster_info[0].items():
                if info["size"] == 2:
                    net_name = name
                    break
            obs = NetworkObserver( calc=calc, cluster_name=net_name, element="Mg" )

            # Several hard coded tests
            calc.update_cf( (0,"Al","Mg") )
            size = 2
            clusters = ceBulk.cluster_info[0][net_name]["indices"]
            for sub_clust in clusters:
                calc.update_cf( (sub_clust[0],"Al","Mg") )
                # Call the observer
                obs(None)

                res = obs.fast_cluster_tracker.get_cluster_statistics_python()
                self.assertEqual( res["cluster_sizes"][0], size )
                size += 1
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
