from cemc.mcmc import SGCMonteCarlo
import numpy as np
import time


class SGCMonteCarloWithGivenSite(SGCMonteCarlo):
    def __init__(self, atoms, T, **kwargs):
        SGCMonteCarlo.__init__(self, atoms, T, **kwargs)

    def _get_trial_move(self, site):
        """Create one trial move."""
        old_symb = self.atoms[site].symbol

        new_symb = old_symb
        n = len(self.symbols)
        while new_symb == old_symb:
            new_symb = self.symbols[np.random.randint(low=0, high=n)]
        system_changes = [(site, old_symb, new_symb)]
        return system_changes


class DamageSpreadingMC(object):
    def __init__(self, atoms1, atoms2, T, **kwargs):
        self.natoms = len(atoms1)
        self.track_time = 10 * self.natoms
        self.orig_atoms1 = atoms1.copy()
        self.orig_atoms2 = atoms2.copy()
        if "track_time" in kwargs.keys():
            self.track_time = kwargs.pop("track_time")

        self.sgc1 = SGCMonteCarloWithGivenSite(atoms1, T, **kwargs)
        self.sgc2 = SGCMonteCarloWithGivenSite(atoms2, T, **kwargs)
        self.damage = np.zeros(self.track_time)
        self.damage[0] = np.sum(atoms1.numbers != atoms2.numbers)
        self.current_time = 1
        self.current_num_different = self.damage[0]
        self.output_every = 30

    def _get_trial_moves(self):
        """Generate one trial move."""
        site = np.random.randint(0, self.natoms)
        symb1 = self.sgc1.atoms[site].symbol
        symb2 = self.sgc2.atoms[site].symbol
        is_different = symb1 != symb2

        # Find a new symbol to insert
        syst_change1 = self.sgc1._get_trial_move(site)
        if is_different:
            syst_change2 = self.sgc2._get_trial_move(site)
        else:
            syst_change2 = syst_change1
        return syst_change1, syst_change2

    def _update_systems(self):
        """Update the systems."""
        ch1, ch2 = self._get_trial_moves()
        acc1 = self.sgc1._accept(ch1)
        acc2 = self.sgc2._accept(ch2)

        # Compare the old symbols
        was_different = ch1[0][1] != ch2[0][1]
        if acc1:
            self.sgc1.atoms.get_calculator().clear_history()
        else:
            self.sgc1.atoms.get_calculator().undo_changes()

        if acc2:
            self.sgc2.atoms.get_calculator().clear_history()
        else:
            self.sgc2.atoms.get_calculator().undo_changes()

        # Check if the system are equal or different
        indx = ch1[0][0]
        symb1 = self.sgc1.atoms[indx].symbol
        symb2 = self.sgc2.atoms[indx].symbol
        is_different = symb1 != symb2
        if was_different and not is_different:
            self.current_num_different -= 1
        elif not was_different and is_different:
            self.current_num_different += 1
        self.damage[self.current_time] = self.current_num_different
        self.current_time += 1

    def _log(self, msg):
        print(msg)

    def reset(self):
        """Reset the calculation to enable"""
        self.current_time = 1
        self.damage[1:] = 0.0
        self.sgc1.reset()
        self.sgc2.reset()

        orig_symb1 = [a.symbol for a in self.orig_atoms1]
        orig_symb2 = [a.symbol for a in self.orig_atoms2]
        self.sgc1.atoms.get_calculator().set_symbols(orig_symb1)
        self.sgc2.atoms.get_calculator().set_symbols(orig_symb2)

    def runMC(self, chem_pot):
        """Run damage spreading Monte Carlo."""
        now = time.time()
        self.sgc1.chemical_potential = chem_pot
        self.sgc2.chemical_potential = chem_pot
        for _ in range(self.track_time - 1):
            if time.time() - now > self.output_every:
                self._log("On step {} of {}".format(self.current_time,
                                                    self.track_time))
                now = time.time()
            self._update_systems()

        return self.damage
