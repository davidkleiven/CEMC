from cemc.mcmc import NucleationSampler
from cemc.mcmc.nucleation_sampler import Mode
from cemc.mcmc import Montecarlo
from cemc.mcmc import NetworkObserver
import numpy as np
from ase.visualize import view


class CanonicalNucleationMC(Montecarlo):
    def __init__(self, atoms, T, **kwargs):
        self.nuc_sampler = kwargs.pop("nucleation_sampler")
        self.network_name = kwargs.pop("network_name")
        self.network_element = kwargs.pop("network_element")
        conc = kwargs.pop("concentration")

        super(CanonicalNucleationMC, self).__init__(atoms, T, **kwargs)
        self.n_atoms = {key: int(value*len(self.atoms)) for key, value
                        in conc.items()}
        self.network = NetworkObserver(calc=self.atoms._calc,
                                       cluster_name=self.network_name,
                                       element=self.network_element)
        self.attach(self.network)

    def _accept(self, system_changes):
        move_accepted = Montecarlo._accept(self, system_changes)
        in_window = self.nuc_sampler.is_in_window(self.network)
        return move_accepted and in_window

    def _get_trial_move(self):
        """Perform a trial move."""
        if not self.nuc_sampler.is_in_window(self.network):
            self.network(None)
            msg = "System is outside the window before the trial move "
            msg += "is performed!\n"
            raise RuntimeError(msg)
            #stat = self.network.get_statistics()
            #num_clusters = stat["number_of_clusters"]
            #max_size = stat["max_size"]
            #lower,upper = self.nuc_sampler.get_window_boundaries(self.nuc_sampler.current_window)
            #msg += "Num. clusters: {}. Max size: {}. Window limits: [{},{})".format(num_clusters,max_size,lower,upper)
        return Montecarlo._get_trial_move(self)

    def get_atoms_count(self):
        """
        Get statistics of the atoms count
        """
        count = {}
        for atom in self.atoms:
            if atom.symbol not in count.keys():
                count[atom.symbol] = 1
            else:
                count[atom.symbol] += 1
        return count

    def sort_natoms(self):
        """
        Sort the number of atoms dictionary such that the lowest enters first
        """
        tuple_natoms = [(value, key) for key, value in self.n_atoms.items()]
        tuple_natoms.sort()
        return tuple_natoms

    def add_additional_atoms_to_match_concentration(self):
        """
        After there might be too few solute atoms after the initial cluster
        have been generated
        """
        atoms_count = self.get_atoms_count()
        sorted_natoms = self.sort_natoms()
        major_element = sorted_natoms[-1][1]
        max_trial_attemps = 100 * len(self.atoms)
        self.atoms._calc.clear_history()
        for i in range(0, len(sorted_natoms) - 1):
            symb = sorted_natoms[i][1]
            at_count = atoms_count[symb]
            diff = self.n_atoms[symb] - atoms_count[symb]
            if diff < 0:
                msg = "The system contains two many solute atoms. "
                msg += "That should not be possible!\n"
                msg += "Failing for atom {}. ".format(symb)
                msg += "Number present: {}. ".format(at_count)
                msg += "Concentration requires: {}".format(self.n_atoms[symb])
                raise RuntimeError(msg)

            num_inserted = 0
            num_trials = 0
            while(num_inserted < diff and num_trials < max_trial_attemps):
                indx = np.random.randint(low=0, high=len(self.atoms))
                num_trials += 1
                if self.atoms[indx].symbol == major_element:
                    syst_change = (indx, self.atoms[indx].symbol, symb)
                    self.atoms._calc.update_cf(syst_change)
                    if self.nuc_sampler.is_in_window(self.network):
                        num_inserted += 1
                        self.atoms._calc.clear_history()
                    else:
                        self.atoms._calc.undo_changes()
            if num_trials >= max_trial_attemps:
                msg = "Did not manage to create a system with the correct "
                msg += "composition!"
                raise RuntimeError(msg)

        # Make sure that the system has the correct composition
        count = self.get_atoms_count()
        for key in count.keys():
            if count[key] != self.n_atoms[key]:
                msg = "The system appears to have wrong composition.\n"
                msg += "Current atoms count: {}. ".format(count)
                msg += "Required atoms count: {}".format(self.n_atoms)
                raise RuntimeError(msg)

        # Re-initialize the tracker
        self._build_atoms_list()
        self.atoms._calc.clear_history()

    def run(self, nsteps=10000):
        """
        Run samples in each window until a desired precission is found
        """

        n_solute_atoms = self.n_atoms[self.network_element]

        for i in range(self.nuc_sampler.n_windows):
            self.log("Window {} of {}".format(i, self.nuc_sampler.n_windows))
            self.nuc_sampler.current_window = i

            lower, upper = self.nuc_sampler._get_window_boundaries(i)
            if 0.5*(lower+upper) >= n_solute_atoms:
                msg = "The concentration of solute atoms is too low to "
                msg += "simulate this size. \n Aborting."
                self.log(msg)
                break
            self.reset()
            self.nuc_sampler.bring_system_into_window(self.network)
            self.add_additional_atoms_to_match_concentration()

            self.nuc_sampler.mode = Mode.equillibriate
            self._estimate_correlation_time()
            self._equillibriate()
            self.nuc_sampler.mode = Mode.sample_in_window

            current_step = 0
            while current_step < nsteps:
                current_step += 1
                self._mc_step()
                self.nuc_sampler.update_histogram(self)
                self.network.reset()
