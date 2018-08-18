from ase.calculators.calculator import Calculator
from ase.ce.corrFunc import CorrFunction
from ase.ce import BulkCrystal
from ase.ce import BulkSpacegroup
from ase.build import bulk
import unittest
from itertools import product, combinations
import os
import numpy as np
import copy
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plt
from ase.visualize import view
from cemc.mcmc import linear_vib_correction as lvc
from inspect import getargspec
from cemc.mcmc.util import trans_matrix2listdict
import gc

from mpi4py import MPI
try:
    from cemc.ce_updater import ce_updater as ce_updater
    use_cpp = True
except Exception as exc:
    use_cpp = False
    print(str(exc))
    print("Could not find C++ version, falling back to Python version")


def get_max_dia_name():
    """
    In the past max_cluster_dist was named max_cluster_dia.
    We support both version here
    """
    args = getargspec(BulkCrystal.__init__).args
    if "max_cluster_dia" in args:
        return "max_cluster_dia"
    return "max_cluster_dist"


def get_ce_calc(small_bc, bc_kwargs, eci=None, size=[1, 1, 1],
                db_name="temp_db.db"):
    """
    Constructs a CE calculator for a supercell by first computing the correlation function
    from a small cell

    :param small_bc: Instance of BulkCrystal or BulkSpacegroup with a relatively small unitcell
    :param bc_kwargs: dictionary of the keyword arguments used to construct small_bc
    :param eci: Effective Cluster Interactions
    :param size: The atoms in small_bc will be extended by this amount
    :param db_name: Database to store info in for the large cell
    """
    nproc = MPI.COMM_WORLD.Get_size()
    unknown_type = False
    large_bc = small_bc
    init_cf = None
    error_happened = False
    msg = ""

    if not os.path.exists(db_name) and nproc > 1:
        raise IOError("The database has to be prepared prior to calling "
                      "get_ce_calc")
    try:
        max_size_eci = get_max_size_eci(eci)
        if "max_cluster_dia" in bc_kwargs.keys():
            if max_size_eci > bc_kwargs["max_cluster_dia"]:
                msg = "ECI specifies a cluster size larger than "
                msg += "ClusterExpansionSetting tracks!"
                raise ValueError(msg)
            print("Initializing calculator with small BC")

        calc1 = CE(small_bc, eci)
        print("Initialization finished")
        init_cf = calc1.get_cf()
        min_length = small_bc.max_cluster_dia
        bc_kwargs["size"] = size
        size_name = get_max_dia_name()
        bc_kwargs[size_name] = min_length
        bc_kwargs["db_name"] = db_name

        if isinstance(small_bc, BulkCrystal):
            large_bc = BulkCrystal(**bc_kwargs)
        elif isinstance(small_bc, BulkSpacegroup):
            large_bc = BulkSpacegroup(**bc_kwargs)
        else:
            unknown_type = True
    except MemoryError:
        # No default error message here
        error_happened = True
        msg = "Memory Error. Most likely went out "
        msg += " of memory when initializing an array"
    except Exception as exc:
        error_happened = True
        msg = str(exc)
        print(msg)

    # Broad cast the error flag and raise error on all processes
    error_happened = MPI.COMM_WORLD.allreduce(error_happened)
    all_msg = MPI.COMM_WORLD.allgather(msg)
    for item in all_msg:
        if item != "":
            msg = item
            break

    if error_happened:
        raise RuntimeError(msg)

    unknown_type = MPI.COMM_WORLD.bcast(unknown_type, root=0)
    if unknown_type:
        msg = "The small_bc argument has to by of type "
        msg += "BulkCrystal or BulkSpacegroup"
        raise TypeError(msg)
    # large_bc = MPI.COMM_WORLD.bcast(large_bc, root=0)
    # init_cf = MPI.COMM_WORLD.bcast(init_cf, root=0)
    calc2 = CE(large_bc, eci, initial_cf=init_cf)
    return calc2


def get_max_size_eci(eci):
    """Finds the maximum cluster name given in the ECIs."""
    max_size = 0
    for key in eci.keys():
        size = int(key[1])
        if size > max_size:
            max_size = size
    return max_size


class CE(Calculator):
    """
    Class for updating the CE when symbols change

    :param BC: Instance of BulkCrystal or BulkSpacegroup from ASE
    :param eci: Dictionary with the effective cluster interactions
    :param initial_cf: Dictionary with the correlation function of the atoms
        object in BC
    """

    implemented_properties = ["energy"]

    def __init__(self, BC, eci=None, initial_cf=None):
        Calculator.__init__(self)
        self.BC = BC
        self.corrFunc = CorrFunction(self.BC)
        cf_names = list(eci.keys())
        if initial_cf is None:
            msg = "Calculating {} correlation ".format(len(cf_names))
            msg += "functions from scratch"
            print(msg)
            self.cf = self.corrFunc.get_cf_by_cluster_names(self.BC.atoms,
                                                            cf_names)
        else:
            self.cf = initial_cf
        print("Correlation functions initialized...")

        self.eci = eci

        self.atoms = self.BC.atoms

        # Keep a copy of the original symbols
        symbols = [atom.symbol for atom in self.BC.atoms]
        self._check_trans_mat_dimensions()

        self.old_cfs = []
        self.old_atoms = self.atoms.copy()
        self.changes = []
        self.ctype = {}
        # self.convert_cluster_indx_to_list()

        if isinstance(self.BC.trans_matrix, np.ndarray):
            self.BC.trans_matrix = np.array(
                self.BC.trans_matrix).astype(
                np.int32)

        self.updater = None
        if use_cpp:
            print("Initializing C++ calculator...")
            self.updater = ce_updater.CEUpdater()
            self.updater.init(self.BC, self.cf, self.eci)
            print("C++ module initialized...")

            if (not self.updater.ok()):
                raise RuntimeError("Could not initialize C++ CE updater")

        if use_cpp:
            self.clear_history = self.updater.clear_history
            self.undo_changes = self.updater.undo_changes
            self.update_cf = self.updater.update_cf
        else:
            raise ImportError("Could not find C++ backend for the updater!")

        # Set the symbols back to their original value
        self.set_symbols(symbols)
        self._linear_vib_correction = None

    def _check_trans_mat_dimensions(self):
        """
        Check that dimension of the trans matrix matches the number of atoms
        """
        if isinstance(self.BC.trans_matrix, list):
            n_sites = len(self.BC.trans_matrix)
        elif isinstance(self.BC.trans_matrix, np.ndarray):
            n_sites = self.BC.trans_matrix.shape[0]

        # Make sure that the database information fits
        if len(self.BC.atoms) != n_sites:
            msg = "The number of atoms and the dimension of the translation "
            msg += "matrix is inconsistent\n"
            msg += "Num atoms: {}. ".format(len(self.BC.atoms))
            msg += "Num row trans mat: {}".format(n_sites)
            raise ValueError(msg)

    def get_full_cluster_names(self, cnames):
        """
        Returns the full cluster names with decoration info in the end

        :param cnames: List of the current cluster names
        """
        full_names = self.cf.keys()
        only_prefix = [name.rpartition("_")[0] for name in full_names]
        full_cluster_names = []

        # First insert the one body names, nothing to be done for them
        for name in cnames:
            if name.startswith("c1"):
                full_cluster_names.append(name)
        for name in cnames:
            if name.startswith("c1"):
                continue
            indx = only_prefix.index(name)
            full_cluster_names.append(full_names[indx])
        return full_cluster_names

    @property
    def linear_vib_correction(self):
        return self._linear_vib_correction

    @linear_vib_correction.setter
    def linear_vib_correction(self, linvib):
        if not isinstance(linvib, lvc.LinearVibCorrection):
            raise TypeError(
                "Linear vib correction has to be of type LinearVibCorrection!")
        if (self.linear_vib_correction is not None):
            orig_eci = self.linear_vib_correction.reset(self.eci)
            if (orig_eci is not None):
                self.eci = orig_eci
            self.update_ecis(self.eci)
        self._linear_vib_correction = linvib
        if self.updater is not None:
            # This just initialize a LinearVibCorrection object, it does not
            # change the ECIs
            self.updater.add_linear_vib_correction(
                ce_updater.map_str_dbl(linvib.eci_per_kbT))

    def include_linvib_in_ecis(self, T):
        """
        Includes the effect of linear vibration correction in the ECIs

        :param T: Temperature in Kelvin
        """
        if (self.linear_vib_correction is None):
            return

        # Reset the ECIs to the original
        self.update_ecis(self.eci)
        self.ecis = self.linear_vib_correction.include(self.eci, T)
        self.update_ecis(self.eci)

    def vib_energy(self, T):
        """
        Returns the vibration energy per atom

        :param T: Temperature in kelving
        """
        if (self.updater is not None):
            return self.updater.vib_energy(T)

        if (self.linear_vib_correction is not None):
            return self.linear_vib_correction.energy(T, self.cf)
        return 0.0

    def get_energy(self):
        """
        Returns the energy of the system
        """
        energy = 0.0
        if (self.updater is None):
            for key, value in self.eci.items():
                energy += value * self.cf[key]
            energy *= len(self.atoms)
        else:
            energy = self.updater.get_energy()
        return energy

    def create_ctype_lookup(self):
        """
        Creates a lookup table for cluster types based on the prefix
        """
        for n in range(2, len(self.BC.cluster_names)):
            for ctype in range(len(self.BC.cluster_names[n])):
                name = self.BC.cluster_names[n][ctype]
                prefix = name  # name.rpartition('_')[0]
                self.ctype[prefix] = (n, ctype)

    def calculate(self, atoms, properties, system_changes):
        """
        Calculates the energy. The system_changes is assumed to be a list
        of tuples of the form (indx,old_symb,new_symb)

        :param atoms: Atoms object. Note that this is not used
            to fit the signature of this function in ASE. The energy returned,
            is the one of the internal atoms object *after* system_changes is
            applied
        :param properties: Has to be ["energy"]
        :param system_changes: Updates to the system. Same signature as
            :py:meth:`cemc.mcmc.MCObserver.__call__`
        """
        energy = self.updater.calculate(system_changes)
        self.cf = self.updater.get_cf()
        self.results["energy"] = energy
        return self.results["energy"]

    def get_cf(self):
        """
        Returns the correlation functions
        """
        if (self.updater is None):
            return self.cf
        else:
            return self.updater.get_cf()

    def update_ecis(self, new_ecis):
        """
        Updates the ecis

        :param new_ecis: New ECI values
        """
        self.eci = new_ecis
        if (self.updater is not None):
            self.updater.set_ecis(self.eci)

    def get_singlets(self, array=None):
        """
        Return the singlets
        """
        if self.updater is None:
            singlets = []
            for key, value in self.cf.items():
                if (key.startswith("c1")):
                    singlets.append(value)
            array = np.array(singlets)
            return array
        else:
            if array is None:
                return self.updater.get_singlets()
            return self.updater.get_singlets(array)

    def set_composition(self, comp):
        """
        Change composition of an object.

        :param comp: Dictionary with the new composition. If you want
            to set the composition to for instance 20%% Mg and 80 %% Al, this
            argument should be {"Mg":0.2, "Al":0.8}
        """
        # Verify that the sum of the compositions is one
        tot_conc = 0.0
        max_element = None
        max_conc = 0.0
        for key, conc in comp.items():
            tot_conc += conc
            if (conc > max_conc):
                max_element = key
                max_conc = conc

        if (np.abs(tot_conc - 1.0) > 1E-6):
            raise ValueError("The specified concentration does not sum to 1!")

        # Change all atoms to the one with the highest concentration
        init_elm = max_element
        for i in range(len(self.atoms)):
            # self.update_cf( (i,self.atoms[i].symbol,init_elm) ) # Set all
            # atoms to init element
            self.calculate(
                self.atoms, ["energy"], [
                    (i, self.atoms[i].symbol, init_elm)])
        start = 0
        for elm, conc in comp.items():
            if (elm == init_elm):
                continue
            n_at = int(round(conc * len(self.atoms)))
            for i in range(start, start + n_at):
                self.update_cf((i, init_elm, elm))
            start += n_at
        self.clear_history()

    def set_symbols(self, symbs):
        """
        Change the symbols of the entire atoms object

        :param symbs: List of new symbols
        """
        if (len(symbs) != len(self.atoms)):
            raise ValueError(
                "Length of the symbols array has to match"
                "the length of the atoms object.!")
        for i, symb in enumerate(symbs):
            self.update_cf((i, self.atoms[i].symbol, symb))
        self.clear_history()

    def singlet2comp(self, singlets):
        """
        Convert singlet to compositions

        :param singlets: Singlet values
        """
        bfs = self.BC.basis_functions

        if (len(singlets.keys()) != len(bfs)):
            msg = "The number singlet terms specified is different "
            msg += "from the number of basis functions\n"
            msg += "Given singlet terms: {}\n".format(singlets)
            msg += "Basis functions: {}\n".format(bfs)
            raise ValueError(msg)

        # Generate system of equations
        rhs = np.zeros(len(bfs))
        # Concentration of this element is implicitly determined via the others
        spec_element = list(bfs[0].keys())[0]
        for key, value in singlets.items():
            dec = int(key[-1])
            rhs[dec] = value - bfs[dec][spec_element]

        matrix = np.zeros((len(bfs), len(bfs)))
        for key in singlets.keys():
            row = int(key[-1])
            col = 0
            for element in bfs[0].keys():
                if (element == spec_element):
                    continue
                matrix[row, col] = bfs[row][element] - bfs[row][spec_element]
                col += 1
        concs = np.linalg.solve(matrix, rhs)

        eps = 1E-6
        # Allow some uncertainty in the solution
        for i in range(len(concs)):
            if (concs[i] < 0.0 and concs[i] > -eps):
                concs[i] = 0.0
        conc_spec_element = 1.0 - np.sum(concs)

        if (conc_spec_element < 0.0 and conc_spec_element > -eps):
            conc_spec_element = 0.0

        # Some trivial checks
        if (conc_spec_element > 1.0 or conc_spec_element < 0.0):
            msg = "Something strange happened when converting singlets to "
            msg += "composition\n"
            msg += "Concentration of one of the implicitly "
            msg += "determined element is {}".format(conc_spec_element)
            raise RuntimeError(msg)

        if (np.any(concs > 1.0) or np.any(concs < 0.0)):
            msg = "Something went wrong when the linear system of "
            msg += "equations were solved.\n"
            msg += "Final concentration is {}".format(concs)
            raise RuntimeError(msg)

        conc_dict = {}
        counter = 0
        for element in bfs[0].keys():
            if (element == spec_element):
                conc_dict[element] = conc_spec_element
            else:
                conc_dict[element] = concs[counter]
                counter += 1
        return conc_dict

    def set_singlets(self, singlets):
        """
        Brings the system into a certain configuration such that the singlet
        has a certain value. Note that depending on the size of atoms,
        it is not all singlet value that are possible. So the composition
        obtained in the end, may differ slightly from the intended value.

        :param singlets: Singlet values
        """
        conc = self.singlet2comp(singlets)
        self.set_composition(conc)

    def write(self, fname):
        """
        Stores all nessecary information required to restart the calculation
        from the state it ended
        """
        backup_data = {}
        backup_data["cf"] = self.get_cf()
