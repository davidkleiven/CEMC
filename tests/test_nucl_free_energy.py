import unittest
import os
try:
    from cemc.mcmc import NucleationSampler, SGCNucleation
    from cemc.mcmc import CanonicalNucleationMC, FixedNucleusMC
    from ase.clease import CEBulk, Concentration
    from ase.clease import CorrFunction
    from cemc.mcmc import InertiaCrdInitializer
    from cemc import CE
    from cemc import get_atoms_with_ce_calc
    import numpy as np
    from helper_functions import get_ternary_BC, get_example_ecis
    available = True
except Exception as exc:
    print(str(exc))
    available = False


def get_network_name(cnames):
    for name in cnames:
        if int(name[1]) == 2:
            return name
    raise RuntimeError("No pair cluster found!")


class TestNuclFreeEnergy( unittest.TestCase ):
    def test_no_throw(self):
        if not available:
            self.skipTest("ASE version does not have CE!")
            return
        no_throw = True
        msg = ""
        try:
            db_name = "temp_nuc_db.db"
            if os.path.exists(db_name):
                os.remove(db_name)
            conc = Concentration(basis_elements=[["Al", "Mg"]])
            kwargs = {
                "crystalstructure": "fcc", "a": 4.05,
                "size": [3, 3, 3],
                "concentration": conc, "db_name": db_name,
                "max_cluster_size": 3
            }
            ceBulk = CEBulk(**kwargs)
            ceBulk.reconfigure_settings()
            cf = CorrFunction(ceBulk)
            cf = cf.get_cf(ceBulk.atoms)

            ecis = {key: 0.001 for key in cf.keys()}
            calc = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[5, 5, 5],
                               db_name="sc5x5x5.db")
            ceBulk = calc.BC
            ceBulk.atoms.set_calculator(calc)

            chem_pot = {"c1_0": -1.0651526881167124}
            sampler = NucleationSampler(
                size_window_width=10,
                chemical_potential=chem_pot, max_cluster_size=20,
                merge_strategy="normalize_overlap")

            nn_name = get_network_name(ceBulk.cluster_family_names_by_size)

            mc = SGCNucleation(
                ceBulk.atoms, 30000, nucleation_sampler=sampler,
                network_name=[nn_name],  network_element=["Mg"],
                symbols=["Al", "Mg"], chem_pot=chem_pot)
            mc.runMC(steps=2)
            sampler.save(fname="test_nucl.h5")

            mc = CanonicalNucleationMC(
                ceBulk.atoms, 300, nucleation_sampler=sampler,
                network_name=[nn_name],  network_element=["Mg"],
                concentration={"Al": 0.8, "Mg": 0.2}
                )
            symbs = [atom.symbol for atom in ceBulk.atoms]
            symbs[0] = "Mg"
            symbs[1] = "Mg"
            mc.set_symbols(symbs)
            mc.runMC(steps=2)
            sampler.save(fname="test_nucl_canonical.h5")
            elements = {"Mg": 6}
            calc.set_composition({"Al": 1.0, "Mg": 0.0})
            mc = FixedNucleusMC(ceBulk.atoms, 300,
                                network_name=[nn_name], network_element=["Mg"])
            mc.insert_symbol_random_places("Mg", num=1, swap_symbs=["Al"])
            mc.runMC(steps=2, elements=elements, init_cluster=True)
            os.remove("sc5x5x5.db")
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue(no_throw, msg=msg)

    def _spherical_nano_particle_matches(self, conc_init):
        # Construct a spherical nano particle
        from ase.cluster.cubic import FaceCenteredCubic
        from random import choice
        surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        layers = [6, 9, 5]
        lc = 4.05
        nanoparticle = FaceCenteredCubic('Al', surfaces, layers,
                                         latticeconstant=lc)
        nanoparticle.info = {"lc": lc}
        symbs = ["Mg", "Si"]
        for atom in nanoparticle:
            atom.symbol = choice(symbs)
        inert1, inert2 = self._nano_particle_matches(conc_init, nanoparticle)
        match, msg = self._two_inertia_tensors_matches(inert1, inert2)
        return match, msg

    def _two_inertia_tensors_matches(self, inert1, inert2):
        if not np.allclose(inert1, inert2, rtol=1E-4):
            msg = "Inertia tensor does not match\n"
            msg += "Original: {}\n".format(inert1)
            msg += "Calculated from bulk: {}\n".format(inert2)
            return False, msg
        return True, ""

    def _nano_particle_matches(self, conc_init, nanoparticle):
        """Check the inertial tensor of a fixed nano particle is conserved."""
        # Test some inertia calculation
        from ase.build import bulk
        from itertools import product
        from ase.clease.tools import wrap_and_sort_by_position

        lc = nanoparticle.info["lc"]
        pos = nanoparticle.get_positions()
        com = np.sum(pos, axis=0) / pos.shape[0]
        nanoparticle.translate(-com)
        pos = nanoparticle.get_positions()
        orig_inertia = np.zeros((3, 3))
        for comb in product(list(range(3)), repeat=2):
            i1 = comb[0]
            i2 = comb[1]
            orig_inertia[i1, i2] = np.sum(pos[:, i1] * pos[:, i2])

        if not np.allclose(orig_inertia, orig_inertia.T):
            raise ValueError("Intertia tensor of nano particle is not "
                             "symmetric!")
        orig_principal = np.linalg.eigvalsh(orig_inertia)

        # Now find the atom in the nanopartorig_principalicle closest to the origin
        # and put it exactly at the origin
        lengths = np.sum(pos**2, axis=1)
        indx = np.argmin(lengths)
        nanoparticle.translate(-nanoparticle[indx].position)
        blk = bulk("Al", crystalstructure="fcc", a=lc)

        blk = blk * (20, 20, 20)
        blk = wrap_and_sort_by_position(blk)
        cell = blk.get_cell()
        diag = 0.5 * (cell[0, :] + cell[1, :] + cell[2, :])
        nanoparticle.translate(diag)
        pos = nanoparticle.get_positions()

        # Inser the nano particle
        pos_blk = blk.get_positions()
        used_sites = []

        for i, atom in enumerate(nanoparticle):
            diff = pos_blk - pos[i, :]
            lengths = np.sqrt(np.sum(diff**2, axis=1))
            indx = np.argmin(lengths)
            if indx in used_sites:
                raise RuntimeError("Two symbols have the same closest atom!")
            blk[indx].symbol = atom.symbol
            used_sites.append(indx)

        # Apply some translation
        trans = 0.2 * diag
        blk.translate(trans)
        blk = wrap_and_sort_by_position(blk)

        # Manually alter the atoms object and rebuild the atoms list
        conc_init.inert_obs.atoms = blk
        conc_init.inert_obs.pos = blk.get_positions()

        # Inertia tensor
        conc_init.inert_obs.init_com_and_inertia()
        inertia_tens = conc_init.principal_inertia(None, [])
        return inertia_tens, orig_principal


    def test_with_inertia_reac_crd(self):
        if not available:
            self.skipTest("ASE version does not have CE!")

        msg = ""
        no_throw = True
        try:
            bc = get_ternary_BC()
            ecis = get_example_ecis(bc=bc)
            calc = CE(bc, eci=ecis)
            bc.atoms.set_calculator(calc)

            T = 200
            nn_names = [name for name in bc.cluster_family_names
                        if int(name[1]) == 2]

            mc = FixedNucleusMC(
                bc.atoms, T, network_name=nn_names,
                network_element=["Mg", "Si"])

            elements = {"Mg": 4, "Si": 4}
            mc.insert_symbol_random_places("Mg", num=1, swap_symbs=["Al"])
            mc.grow_cluster(elements)
            conc_init = InertiaCrdInitializer(
                fixed_nucl_mc=mc, matrix_element="Al",
                cluster_elements=["Mg", "Si"])
            
            mc.runMC(steps=100, init_cluster=False)

            match, match_msg = self._spherical_nano_particle_matches(conc_init)
            self.assertTrue(match, msg=match_msg)
        except Exception as exc:
            no_throw = False
            msg = str(exc)

        self.assertTrue(no_throw, msg=msg)

    def test_inertia_observer(self):
        """Test the inertia observer."""
        if not available:
            self.skipTest("ASE version does not have CE!")

        msg = ""
        no_throw = True
        try:
            from cemc.mcmc import FixEdgeLayers
            from cemc.mcmc import InertiaTensorObserver
            bc, args = get_ternary_BC(ret_args=True)
            ecis = get_example_ecis(bc=bc)
            calc = get_atoms_with_ce_calc(bc, args, eci=ecis, size=[8, 8, 8], db_name="inertia_obs.db")
            bc = calc.BC
            bc.atoms.set_calculator(calc)

            T = 200
            nn_names = [name for name in bc.cluster_family_names
                        if int(name[1]) == 2]

            mc = FixedNucleusMC(
                bc.atoms, T, network_name=nn_names,
                network_element=["Mg", "Si"])

            fixed_layers = FixEdgeLayers(atoms=mc.atoms, thickness=3.0)
            mc.add_constraint(fixed_layers)
            elements = {"Mg": 6, "Si": 6}
            mc.insert_symbol_random_places("Mg", num=1, swap_symbs=["Al"])
            mc.grow_cluster(elements)
            
            inert_obs = InertiaTensorObserver(atoms=mc.atoms, cluster_elements=["Mg", "Si"])
            mc.attach(inert_obs)
            for _ in range(10):
                mc.runMC(steps=100, elements=elements, init_cluster=False)

                obs_I = inert_obs.inertia
                indices = []
                for atom in mc.atoms:
                    if atom.symbol in ["Mg", "Si"]:
                        indices.append(atom.index)
                cluster = mc.atoms[indices]
                pos = cluster.get_positions()
                com = np.mean(pos, axis=0)
                pos -= com
                inertia = np.zeros((3, 3))
                for i in range(pos.shape[0]):
                    x = pos[i, :]
                    inertia += np.outer(x, x)
                self.assertTrue(np.allclose(obs_I, inertia))
            os.remove("inertia_obs.db")
        except Exception as exc:
            no_throw = False
            msg = type(exc).__name__ + str(exc)

        self.assertTrue(no_throw, msg=msg)


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
