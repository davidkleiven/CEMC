from cemc.mcmc.nucleation_sampler import Mode
from cemc.mcmc import SGCMonteCarlo
from cemc.mcmc.mc_observers import NetworkObserver
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import TrajectoryWriter
import time
import os
import copy
import json


class DidNotReachProductOrReactantError(Exception):
    def __init__(self, msg):
        super(DidNotReachProductOrReactantError, self).__init__(msg)


class DidNotFindPathError(Exception):
    def __init__(self, msg):
        super(DidNotFindPathError, self).__init__(msg)


class SGCNucleation(SGCMonteCarlo):
    """
    Class to perform Monte Carlo simulations where the main objective is to
    compute the free energy for different network sizes

    See py:class:`cemc.mcmc.SGCMonteCarlo` for parameter explination

    :param NucleationSampler nucleation_sampler: Nucleation Sampler
    :param list network_name: List of clusters that connects constituents of
        a cluster
    :param list network_element: Elements in the atomic cluster
    :param dict chem_pot: The chemical potential at which to perform the
        simulations see :py:meth:`cemc.mcmc.SGCMonteCarlo.runMC`
    :param bool allow_solutes: True/False. If False then all elements
        that can form an atomic cluster has to be located in a cluster
    """

    def __init__(self, atoms, temp, **kwargs):
        self.nuc_sampler = kwargs.pop("nucleation_sampler")
        kwargs["mpicomm"] = None
        self.network_name = kwargs.pop("network_name")
        self.network_element = kwargs.pop("network_element")
        chem_pot = kwargs.pop("chem_pot")
        self.allow_solutes = True
        if "allow_solutes" in kwargs.keys():
            self.allow_solutes = kwargs.pop("allow_solutes")
        super(SGCNucleation, self).__init__(atoms, temp, **kwargs)
        self.chemical_potential = chem_pot

        self.network = NetworkObserver(
            calc=self.atoms._calc, cluster_name=self.network_name,
            element=self.network_element)
        self.network.fixed_num_solutes = False
        self.attach(self.network)

        if self.allow_solutes:
            self.log("Solute atoms in cluster and outside is allowed")
        else:
            self.log("Solute atoms are only allowed in the cluster")

    def _accept(self, system_changes):
        """
        Accept the trial move

        :param list system_changes: See :py:meth:`cemc.mcmc.Montecarlo.accept`

        :return: True/False. If True, the move is accepted
        :rtype: bool
        """
        move_accepted = SGCMonteCarlo._accept(self, system_changes)
        in_window, stat = self.nuc_sampler.is_in_window(
            self.network, retstat=True)
        if not self.allow_solutes:
            new_size = stat["max_size"]
            cur_size = self.nuc_sampler.current_cluster_size
            if new_size != cur_size+1 and new_size != cur_size-1:
                in_window = False
            else:
                self.nuc_sampler.current_cluster_size = new_size
        return move_accepted and in_window

    def _get_trial_move(self):
        """
        Perform a trial move
        """
        if not self.nuc_sampler.is_in_window(self.network):
            raise RuntimeError("System is outside the window before the trial "
                               "move is performed!")
        return SGCMonteCarlo._get_trial_move(self)

    def run(self, nsteps=1000):
        """
        Run samples in each window until a desired precission is found

        :param int nsteps: Number of MC steps in each window
        """
        if self.nuc_sampler.nucleation_mpicomm is not None:
            self.nuc_sampler.nucleation_mpicomm.barrier()

        for i in range(self.nuc_sampler.n_windows):
            self.log("Window {} of {}".format(i, self.nuc_sampler.n_windows))
            self.nuc_sampler.current_window = i
            self.reset()
            self.nuc_sampler.bring_system_into_window(self.network)
            self.current_energy = self.atoms.get_calculator().get_energy()

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

        if self.nuc_sampler.nucleation_mpicomm is not None:
            self.nuc_sampler.nucleation_mpicomm.barrier()

    def remove_snapshot_observers(self):
        """
        Remove all Snapshot observers from the observers
        """
        self.observers = [obs for obs in self.observers
                          if obs.name != "Snapshot"]

    def remove_network_observers(self):
        """
        Remove NetworkObservers
        """
        self.observers = [obs for obs in self.observers
                          if obs[1].name != "NetworkObserver"]

    def is_reactant(self):
        """
        Returns true if the current state is in the reactant region

        :return: True/False. If True, the configuration is classified as a
            reactant.
        :rtype: bool
        """
        if self.max_size_reactant is None:
            raise ValueError("Maximum cluster size to be characterized as "
                             "reactant is not set!")
        stat = self.network.get_statistics()
        return stat["max_size"] < self.max_size_reactant

    def is_product(self):
        """
        Return True if the current state is a product state
        """
        if self.min_size_product is None:
            raise ValueError("Minimum cluster size to be characterized as "
                             "product is not set!")
        stat = self.network.get_statistics()
        return stat["max_size"] >= self.min_size_product

    def _merge_product_and_reactant_path(self, reactant_traj, product_traj,
                                         reactant_symb, product_symb):
        """
        Merge the product and reactant path into one file

        :param str reactant_traj: File with the reactant path
        :param str product_traj: File with the product path
        :param list reactant_symb: List with the reactant symbols
        :param list product_symb: List with the product symbols
        """
        raise NotImplementedError("This function has not been implemented!")

    def _save_list_of_lists(self, fname, data):
        """
        Save a list of lists into a text file

        :param str fname: Filenem
        :param data: List with datapoints
        :type data: list of lists
        """
        with open(fname, 'w') as outfile:
            for sublist in data:
                for entry in sublist:
                    outfile.write("{} ".format(entry))
                outfile.write("\n")

    def reset(self):
        """
        Overrides the parents method
        """
        super(SGCNucleation, self).reset()
        self.network.reset()

    def _read_list_of_lists(self, fname, dtype="str"):
        """
        Read list of lists

        :param str fname: Filename
        :param str dtype: Datatype of the entries in the file
        """
        raise NotImplementedError("Function not implemented!")

    def _symbols2uint(self, symbols, description):
        """
        Convert symbols to description indices.

        Convert array of symbols into a numpy array of indices in the
        desctiption array.

        :param list symbols: List of symbols
        :param list description: List with symbols

        :return: Numpy array with indices
        :rtype: 1D Numpy array of numpy.uint8
        """
        import numpy as np
        nparray = np.zeros(len(symbols), dtype=np.uint8)
        for i, symb in enumerate(symbols):
            nparray[i] = description.index(symb)
        return nparray

    def _uint2symbols(self, nparray, description):
        """
        Convert uint8 array to symbols array

        :param numpy.ndarray nparray: Numpy array with indices
        :param list description: List with symbols

        :return: nparray converted into list with symbols
        :rtype: list of str
        """
        symbs = []
        for i in range(len(nparray)):
            symbs.append(description[nparray[i]])
        return symbs

    def _merge_reference_path(self, res_reactant, res_product):
        """
        This store the reference path into a JSON

        :param dict res_reactant: Results from reactant path
        :param dict res_product: Results from product path

        :return: Combined path
        :rtype: dict
        """
        res_reactant["energy"] = res_reactant["energy"][::-1]
        res_reactant["symbols"] = res_reactant["symbols"][::-1]
        res_reactant["sizes"] = res_reactant["sizes"][::-1]
        combined_path = {}
        combined_path["energy"] = res_reactant["energy"]+res_product["energy"]
        combined_path["symbols"] = res_reactant["symbols"] + \
            res_product["symbols"]
        combined_path["sizes"] = res_reactant["sizes"]+res_product["sizes"]
        return combined_path

    def save_path(self, fname, res):
        """
        Stores the path result to a JSON file

        :param str fname: Filename
        :param dict res: Dictionary with results from path sampling
        """
        res["min_size_product"] = self.min_size_product
        res["max_size_reactant"] = self.max_size_reactant
        with open(fname, 'w') as outfile:
            json.dump(res, outfile)

    def sweep(self, nsteps=None):
        """
        Performs one MC sweep

        :param nsteps: Number of steps in one sweep. If None, the number of
            atoms is used
        :type nsteos: int or None
        """
        if nsteps is None:
            nsteps = len(self.atoms)
        for i in range(nsteps):
            self._mc_step()

    def set_mode(self, mode):
        """
        Set the mode

        :param str mode: New mode
        """
        known_modes = ["bring_system_into_window",
                       "sample_in_window",
                       "equillibriate",
                       "transition_path_sampling"]
        if mode not in known_modes:
            raise ValueError("Mode has to be one of {}".format(known_modes))

        if mode == "bring_system_into_window":
            self.nuc_sampler.mode = Mode.bring_system_into_window
        elif mode == "sample_in_window":
            self.nuc_sampler.mode = Mode.sample_in_window
        elif mode == "equillibriate":
            self.nuc_sampler.mode = Mode.sample_in_window
        elif mode == "transition_path_sampling":
            self.nuc_sampler.mode = Mode.transition_path_sampling

    def set_state(self, symbols):
        """
        Sets the state of the system

        :param list symbols: List of symbols
        """
        self.set_symbols(symbols)
        # self.atoms._calc.set_symbols(symbols)

    def show_statistics(self, path):
        """
        Show a plot indicating if the path is long enough

        :param list path: List with configurations along one path
            Each list is a list of symbols
        """
        import numpy as np
        from matplotlib import pyplot as plt
        product_indicator = []
        reactant_indicator = []
        for state in path:
            self.network.reset()
            self.set_state(state)
            self.network(None)
            if self.is_product():
                product_indicator.append(1)
            else:
                product_indicator.append(0)

            if self.is_reactant():
                reactant_indicator.append(1)
            else:
                reactant_indicator.append(0)
        hB = np.cumsum(product_indicator)
        hA = np.cumsum(reactant_indicator)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(hB, label="Product indicator")
        ax.plot(hA, label="Reactant indicator")
        ax.set_xlabel("MC sweeps")
        ax.legend()

        fig_size = plt.figure()
        ax_size = fig_size.add_subplot(1, 1, 1)
        ax_size.plot(path["sizes"])
        return fig, fig_size

    def find_transition_path(self, initial_cluster_size=None,
                             max_size_reactant=None, min_size_product=None,
                             path_length=1000, max_attempts=100, folder=".",
                             nsteps=None, mpicomm=None):
        """
        Find one transition path

        :param int initial_cluster_size: Initial size of the cluster
        :param int max_size_reactant: Maximum cluster size allowed for a
            configuration to be classified as reactant.
        :param int min_size_product: Minimum cluster size allowed for a
            configuration to be classified as product.
        :param int path_length: Length of a reaction path
        :param int max_attempts: Maximum number of attempts to find a new
            reaction path
        :param str foler: Work directory where results will be stored
        :param nsteps: Number of steps per sweep. If None, the number of
            atoms will be used
        :type nsteps: int or None
        :param mpicomm: MPI communicator object
        :type mpicomm: Intracomm or None
        """
        import numpy as np
        if initial_cluster_size is None:
            raise ValueError("Initial cluster size not given!")
        if max_size_reactant is None:
            raise ValueError("The maximum cluster size allowed for the state "
                             "to be characterized as reactant is not given!")
        if min_size_product is None:
            raise ValueError("The minimum size of cluster allowed for the "
                             "state to be characterized as product is not "
                             "given!")

        trans_path_mpi_comm = mpicomm
        self.mpicomm = None
        self.rank = 0
        size = 1
        if trans_path_mpi_comm is not None:
            self.rank = trans_path_mpi_comm.Get_rank()
            size = trans_path_mpi_comm.Get_size()

        self.log("Running transition path seach on {} processors".format(size))

        self.nuc_sampler.mode = Mode.transition_path_sampling
        self.max_size_reactant = max_size_reactant
        self.min_size_product = min_size_product

        found_reactant_origin = False
        found_product_origin = False
        self.remove_network_observers()
        self.attach(self.network, interval=len(self.atoms))

        num_reactants = 0
        num_products = 0
        default_trajfile = folder+"/default_trajfile{}.traj".format(self.rank)
        reactant_file = folder+"/trajectory_reactant{}.traj".format(self.rank)
        product_file = folder+"/trajectory_product{}.traj".format(self.rank)
        reference_path_file = folder+"/reference_path{}.json".format(self.rank)

        self.network.reset()
        print("Warning! Cluster initialization does not work at the moment!")
        # self.network.grow_cluster( initial_cluster_size )

        init_symbols = [atom.symbol for atom in self.atoms]
        target = "both"
        reactant_res = {}
        product_res = {}

        found_reactant_origin = False
        found_product_origin = False
        for attempt in range(max_attempts):
            self.reset()
            self.atoms._calc.set_symbols(init_symbols)
            res = {"type": "transition_region"}
            try:
                res = self._find_one_transition_path(
                    path_length=path_length/2, trajfile=default_trajfile,
                    target=target, nsteps=nsteps)
            except DidNotReachProductOrReactantError as exc:
                self.log(str(exc))
                self.log("Trying one more time")

            if res["type"] == "reactant":
                num_reactants += 1
                target = "product"  # Reactant is found, search for products
                if not found_reactant_origin:
                    os.rename(default_trajfile, reactant_file)
                    found_reactant_origin = True
                    reactant_res = copy.deepcopy(res)

            elif res["type"] == "product":
                num_products += 1
                target = "reactant"  # Product is found search for reactant
                if not found_product_origin:
                    os.rename(default_trajfile, product_file)
                    found_product_origin = True
                    product_res = copy.deepcopy(res)

            found_path = found_product_origin and found_reactant_origin
            if os.path.exists(default_trajfile):
                os.remove(default_trajfile)

            if found_path:
                combined_path = self._merge_reference_path(
                    reactant_res, product_res)
                combined_path["nsteps_per_sweep"] = nsteps
                self.save_path(reference_path_file, combined_path)
                self.log("Found a path to the product region and a path to "
                         "the reactant region")
                self.log("They are stored in {} and {}"
                         "".format(product_file, reactant_file))
                self.log("The reference path is stored in {}"
                         "".format(reference_path_file))
                # self.show_statistics(combined_path["symbols"])

            # Collect the found_path flag from all processes
            if trans_path_mpi_comm is not None:
                send_buf = np.zeros(1, dtype=np.uint8)
                send_buf[0] = found_path
                recv_buf = np.zeros(1, dtype=np.uint8)
                trans_path_mpi_comm.Allreduce(send_buf, recv_buf)
                found_path = (recv_buf[0] >= 1)

            if found_path:
                break
            self.log("Attempt: {} of {} ended in {} region"
                     "".format(attempt, max_attempts, res["type"]))

        if not found_path:
            msg = "Did not manage to find both a configuration in the product "
            msg += "region and the reactant region\n"
            raise DidNotFindPathError(msg)

    def _add_info_to_atoms(self, atoms):
        """
        Adds info to atoms object

        :param Atoms atoms: Atoms object
        """
        if atoms is None:
            return
        atoms.info["is_product"] = self.is_product()
        atoms.info["is_reactant"] = self.is_reactant()
        atoms.info["cluster_size"] = self.network.max_size

    def _find_one_transition_path(self, path_length=1000,
                                  trajfile="default.traj", target="both",
                                  nsteps=None):
        """
        Finds a transition path by running random samples

        :param int path_length: Number of MC sweep in path
        :param str trajfile: Trajectory file where path will be stored
        :param str target: Target basins (reactant, product or both)
        :param nsteps: Number of MC steps in one sweep
        :type nsteps: int or None
        """
        supported_targets = ["reactant", "product", "both"]
        if target not in supported_targets:
            raise ValueError("Target has to be one of {}"
                             "".format(supported_targets))

        # Check if a snapshot tracker is attached
        traj = TrajectoryWriter(trajfile, mode="w")
        result = {}
        symbs = []
        unique_symbols = []
        for atom in self.atoms:
            if atom.symbol not in unique_symbols:
                unique_symbols.append(atom.symbol)

        output_every_sec = 30
        now = time.time()
        energies = []
        result = {}
        sizes = []
        for sweep in range(int(path_length)):
            self.network.reset()
            if time.time() - now > output_every_sec:
                self.log("Sweep {} of {}".format(sweep, path_length))
                now = time.time()
            self.sweep(nsteps=nsteps)

             # Explicitly enforce a construction of the network
            self.network(None)
            energies.append(self.current_energy)
            symbs.append([atom.symbol for atom in self.atoms])
            atoms = self.network.get_atoms_with_largest_cluster(
                prohibited_symbols=unique_symbols)
            sizes.append(self.network.max_size)
            self._add_info_to_atoms(atoms)
            calc = SinglePointCalculator(atoms, energy=self.current_energy)
            atoms.set_calculator(calc)
            if atoms is None:
                traj.write(self.atoms)
            else:
                traj.write(atoms)

            if target == "reactant":
                if self.is_product():
                    # Terminate before the desired path length is reached
                    result["type"] = "product"
                    result["symbols"] = symbs
                    result["energy"] = energies
                    result["sizes"] = sizes
                    return result
            elif target == "product":
                if self.is_reactant():
                    result["type"] = "reactant"
                    result["symbols"] = symbs
                    result["energy"] = energies
                    result["sizes"] = sizes
                    # Terminate before the desired path length is reached
                    return result

        traj.close()
        if self.is_reactant():
            result["type"] = "reactant"
        elif self.is_product():
            result["type"] = "product"
        else:
            stat = self.network.get_statistics()
            max_size = stat["max_size"]
            msg = "State did not end up in product or reactant region. "
            msg += "Increase the number of sweeps.\n"
            msg += "Max. cluster size {}.".format(max_size)
            msg += "Max cluster size reactants {}".format(
                self.max_size_reactant)
            msg += "Min cluster size products {}".format(self.min_size_product)
            raise DidNotReachProductOrReactantError(msg)

        result["symbols"] = symbs
        result["energy"] = energies
        result["sizes"] = sizes
        return result
