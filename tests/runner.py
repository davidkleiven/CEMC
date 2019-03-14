import unittest
import os
from cemc import TimeLoggingTestRunner
import test_activity_sampler
import test_all_examples
import test_canonical_free_energy
import test_CE_updater
import test_chem_pot_ROI
import test_collective_jump_move
import test_eshelby
import test_free_energy_barrier
import test_free_energy
import test_gs_finder
import test_mc_parameter_sweep
import test_mfa
import test_network_observer
import test_nucl_free_energy
import test_phase_boundary_tracker
import test_sgc_mc
import test_strain_energy
import test_transition_path
import test_wang_landau_init
import test_damage_spreading_mc
import test_pseudo_binary_sgc
import test_parallel_tempering
import test_adaptive_bias_potential
import test_isotropic_strain_energy
import test_khacaturyan
import test_binary_phase_diag
import test_cahn_hilliard
import test_cahn_hilliard_phase_field
import test_polynomial_term
import test_phasefield_poly
import test_coupled_euler
import test_gradient_coeff
import test_chgl
import test_hyper_tangent_bvp
import test_diffraction_updater
import test_concentration_observer
import test_interior_minima

try:
    os.mkdir("data")
except OSError as exc:
    pass
except Exception as exc:
    print(str(exc))

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests
suite.addTests(loader.loadTestsFromModule(test_activity_sampler))
suite.addTests(loader.loadTestsFromModule(test_all_examples))
suite.addTests(loader.loadTestsFromModule(test_canonical_free_energy))
suite.addTests(loader.loadTestsFromModule(test_CE_updater))
suite.addTests(loader.loadTestsFromModule(test_chem_pot_ROI))
suite.addTests(loader.loadTestsFromModule(test_collective_jump_move))
suite.addTests(loader.loadTestsFromModule(test_eshelby))
suite.addTests(loader.loadTestsFromModule(test_free_energy_barrier))
suite.addTests(loader.loadTestsFromModule(test_free_energy))
suite.addTests(loader.loadTestsFromModule(test_mc_parameter_sweep))
suite.addTests(loader.loadTestsFromModule(test_gs_finder))
suite.addTests(loader.loadTestsFromModule(test_mfa))
suite.addTests(loader.loadTestsFromModule(test_network_observer))
suite.addTests(loader.loadTestsFromModule(test_nucl_free_energy))
suite.addTests(loader.loadTestsFromModule(test_phase_boundary_tracker))
suite.addTests(loader.loadTestsFromModule(test_sgc_mc))
suite.addTests(loader.loadTestsFromModule(test_strain_energy))
suite.addTests(loader.loadTestsFromModule(test_transition_path))
suite.addTests(loader.loadTestsFromModule(test_wang_landau_init))
suite.addTests(loader.loadTestsFromModule(test_damage_spreading_mc))
suite.addTests(loader.loadTestsFromModule(test_pseudo_binary_sgc))
suite.addTests(loader.loadTestsFromModule(test_parallel_tempering))
suite.addTests(loader.loadTestsFromModule(test_adaptive_bias_potential))
suite.addTest(loader.loadTestsFromModule(test_isotropic_strain_energy))
suite.addTest(loader.loadTestsFromModule(test_khacaturyan))
suite.addTest(loader.loadTestsFromModule(test_binary_phase_diag))
suite.addTest(loader.loadTestsFromModule(test_cahn_hilliard))
suite.addTest(loader.loadTestsFromModule(test_cahn_hilliard_phase_field))
suite.addTest(loader.loadTestsFromModule(test_polynomial_term))
suite.addTest(loader.loadTestsFromModule(test_phasefield_poly))
suite.addTest(loader.loadTestsFromModule(test_coupled_euler))
suite.addTest(loader.loadTestsFromModule(test_gradient_coeff))
suite.addTest(loader.loadTestsFromModule(test_chgl))
suite.addTest(loader.loadTestsFromModule(test_hyper_tangent_bvp))
suite.addTest(loader.loadTestsFromModule(test_diffraction_updater))
suite.addTest(loader.loadTestsFromModule(test_concentration_observer))
suite.addTest(loader.loadTestsFromModule(test_interior_minima))

runner = TimeLoggingTestRunner()
result = runner.run(suite)

generated_files = ["database_with_dft_structures.db", "temp_db_wanglandau.db",
                   "template_bc.db", "temp_nuc_db.db",
                   "temporary_bcnucleationdb.db", "temporary_db.db",
                   "test_db_bcc.db", "test_db_binary_almg.db",
                   "test_db.db", "test_db_fcc.db", "test_db_fcc_super.db",
                   "test_db_gsfinder.db", "test_db_hcp.db",
                   "test_db_network.db", "test_db_sc.db", "test_db_ternary.db",
                   "test_gs_db.db", "test_phase_boundary.db",
                   "test_singlets.db", "wanglandau_test_init.db",
                   "backup_phase_track.h5", "default_output.h5",
                   "test_nucl_canonical.h5", "test_nucl.h5",
                   "test_phase_boundary.h5", "test_phase_boundary_ternary.h5",
                   "BC_wanglandau_Al62Mg63.pkl",
                   "BC_wanglandau_Al63Mg62.pkl", "demo.traj",
                   "relaxed_path.traj", "subbin.csv",
                   "free_energy_barrier.json", "bin_transfer.csv"]


def clean():
    for fname in generated_files:
        try:
            os.remove(fname)
        except OSError:
            pass
        except Exception as exc:
            print(str(exc))


clean()
