import unittest
import os
try:
    from cemc.mcmc import NetworkObserver
    from cemc import CE
    from ase.clease import CEBulk
    from ase.clease import Concentration
    from ase.clease import CorrFunction
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

            conc = Concentration(basis_elements=[["Al","Mg"]])
            a = 4.05
            ceBulk = CEBulk(
                crystalstructure="fcc", a=a, size=[3, 3, 3],
                concentration=conc,
                db_name=db_name, max_cluster_size=3,
                max_cluster_dia=4.5)
            ceBulk.reconfigure_settings()

            cf = CorrFunction(ceBulk)
            atoms = ceBulk.atoms.copy()
            cf = cf.get_cf(atoms)
            eci = {key:0.001 for key in cf.keys()}
            calc = CE(atoms, ceBulk, eci=eci)
            trans_mat = ceBulk.trans_matrix

            for name, info in ceBulk.cluster_info[0].items():
                if info["size"] == 2:
                    net_name = name
                    break
            obs = NetworkObserver(calc=calc, cluster_name=[net_name], element=["Mg"])

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
            obs.get_indices_of_largest_cluster_with_neighbours()
            os.remove("test_db_network.db")

        except Exception as exc:
            msg = "{}: {}".format(type(exc).__name__, str(exc))
            no_throw = False
        self.assertTrue( no_throw, msg=msg )

    # def test_fast_network_update(self):
    #     if not available:
    #         self.skipTest("ASE version does not have CE!")
    #     from cemc.mcmc import FixedNucleusMC
    #     no_throw = True
    #     msg = ""
    #     try:
    #         db_name = "test_db_network.db"

    #         conc_args = {
    #             "conc_ratio_min_1":[[1,0]],
    #             "conc_ratio_max_1":[[0,1]],
    #         }
    #         a = 4.05
    #         ceBulk = CEBulk(
    #             crystalstructure="fcc", a=a, size=[3, 3, 3],
    #             basis_elements=[["Al","Mg"]], conc_args=conc_args,
    #             db_name=db_name, max_cluster_size=3)
    #         ceBulk.reconfigure_settings()
    #         cf = CorrFunction(ceBulk)
    #         cf = cf.get_cf(ceBulk.atoms)
    #         eci = {key:0.001 for key in cf.keys()}
    #         calc = CE( ceBulk, eci=eci )
    #         ceBulk.atoms.set_calculator(calc)

    #         # Insert some magnesium atoms
    #         for name, info in ceBulk.cluster_info[0].items():
    #             if info["size"] == 2:
    #                 net_name = name
    #                 break

    #         mc = FixedNucleusMC(ceBulk.atoms, 500, network_element=["Mg"], network_name=[net_name])
    #         mc.insert_symbol_random_places("Mg", swap_symbs=["Al"])
    #         elements = {"Mg": 10}
    #         mc.grow_cluster(elements)
    #         obs = NetworkObserver(calc=calc, cluster_name=[net_name], element=["Mg"],
    #                     only_one_cluster=True)
    #         obs.collect_statistics_on_call = False
    #         mc.attach(obs, interval=1)
    #         self.assertTrue(obs.has_minimal_connectivity())
    #         from ase.visualize import view
    #         view(mc.atoms)
    #         for _ in range(10):
    #             mc.runMC(steps=200, init_cluster=False, elements=elements)
    #             stat = obs.get_current_cluster_info()

    #             obs.retrieve_clusters_from_scratch()
    #             stat_new = obs.get_current_cluster_info()
    #             print(stat, stat_new)
    #             self.assertAlmostEqual(stat["max_size"], stat_new["max_size"])
    #             self.assertEqual(stat["cluster_sizes"], stat_new["cluster_sizes"])
    #             self.assertAlmostEqual(stat["number_of_clusters"], stat_new["number_of_clusters"])
    #             self.assertEqual(stat["cluster_sizes"][0], 10)
    #         os.remove("test_db_network.db")
    #     except Exception as exc:
    #         no_throw = False
    #         msg = str(exc)
    #     self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
