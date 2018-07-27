"""
Module holding the nessecary classes for tracing a phase boundary
"""

import os
from itertools import combinations
import logging

import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from mpi4py import MPI
import h5py as h5
from ase.units import kB

from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
from cemc.tools.phase_track_utils import PhaseBoundarySolution
from cemc.tools.phase_track_utils import CECalculators

# With this backend one does not need a screen (useful for clusters)
plt.switch_backend("Agg")

COMM = MPI.COMM_WORLD


class PhaseChangedOnFirstIterationError(Exception):
    """
    Exception that is raised if the one of the states change phase on
    the first iteration
    """
    pass


class PhaseBoundaryTracker(object):
    """
    Class for tracking the phase boundary between two phases
    NOTE: Has only been confirmed to work for binary systems!

    :params ground_state: List of dictionaries containing the following fields
        * *bc* BulkCrystal object
        * *cf* Correlation function of the state of the atoms object in BulkCrystal
        * *eci* Effective Cluster Interactions
    :param logfile: Filename where log messages should be written
    :param max_singlet_change: The maximum amount the singlet
        terms are allowed to change on each step
    """
    def __init__(
            self,
            ground_states,
            logfile="",
            max_singlet_change=0.05,
            backupfile="backup_phase_track.h5"):

        self._ground_states = ground_states
        for ground_state in self._ground_states:
            check_gs_argument(ground_state)

        self._max_singlet_change = max_singlet_change
        self._backupfile = backupfile
        self._sgc_obj = None
        self._integration_direction = "increasing"

        if os.path.exists(self._backupfile):
            os.remove(self._backupfile)

        self._singlet_names = [
            name for name in self._ground_states[0]["cf"].keys() if name.startswith("c1")]

        self._current_backup_indx = 0

        self._logger = logging.getLogger("PhaseBoundaryTracker")
        self._logger.setLevel(logging.INFO)
        self._ref_indicators = None

        if logfile == "":
            comm_handler = logging.StreamHandler()
        else:
            comm_handler = logging.FileHandler(logfile)

        if not self._logger.handlers:
            self._logger.addHandler(comm_handler)


    def _get_gs_energies(self):
        """
        Return the Ground State Energies
        """
        energy = []
        for ground_state in self._ground_states:
            gs_energy = 0.0
            for key in ground_state["eci"].keys():
                gs_energy += ground_state["eci"][key] * ground_state["cf"][key]
            energy.append(len(ground_state["bc"].atoms) * gs_energy)
        return energy

    def _get_init_chem_pot(self):
        """
        Computes the chemical potential at which the two phases coexists
        at zero kelvin
        """
        num_singlets = len(self._ground_states) - 1
        matrix = np.zeros((num_singlets, num_singlets))
        energy_vector = np.zeros(num_singlets)
        gs_energies = self._get_gs_energies()
        for i in range(num_singlets):
            for j in range(num_singlets):
                ref_singlet = self._ground_states[0]["cf"][self._singlet_names[j]]
                singlet = self._ground_states[i + 1]["cf"][self._singlet_names[j]]
                matrix[i, j] = (ref_singlet - singlet)
            energy_ref = gs_energies[0] / len(self._ground_states[0]["bc"].atoms)
            energy = gs_energies[i + 1] / len(self._ground_states[i + 1]["bc"].atoms)
            energy_vector[i] = energy_ref - energy

        mu_boundary = np.linalg.solve(matrix, energy_vector)
        return mu_boundary


    def _log(self, msg, mode="info"):
        """
        Print message for logging
        """
        rank = COMM.Get_rank()
        if rank == 0:
            if mode == "info":
                self._logger.info(msg)
            elif mode == "warning":
                self._logger.warning(msg)

    def _set_integration_direction(self, T0, Tend):
        """Sets the integration direction."""
        if Tend is None:
            # Use the default which is increasing from 0K
            return
        if T0 > Tend:
            self._integration_direction = "decreasing"
        else:
            self._integration_direction = "increasing"

    def _reached_temperature_end_point(self, T, Tend):
        """Returns true if we reached the temperature end point."""
        if Tend is None:
            # End point not give
            return False

        if self._integration_direction == "increasing":
            if T > Tend:
                return True
        elif self._integration_direction == "decreasing":
            if T < Tend:
                return True
        return False

    def _backup(self, data, dsetname="data"):
        """
        Stores backup data to hdf5 file

        :param data: Dictionary of data to be backed up
        :param dsetname: Basename for all datasets in the h5 file
        """
        rank = COMM.Get_rank()
        if rank == 0:
            with h5.File(self._backupfile, 'a') as hfile:
                grp = hfile.create_group(
                    dsetname +
                    "{}".format(
                        self._current_backup_indx))
                for key, value in data.items():
                    if value is None:
                        continue
                    if key == "images":
                        for img_num, img in enumerate(value):
                            if img is None:
                                continue
                            #img = img.T
                            dset = grp.create_dataset(
                                "img_{}".format(img_num), data=img)
                            dset.attrs['CLASS'] = "IMAGE"
                            dset.attrs['IMAGE_VERSION'] = '1.2'
                            dset.attrs['IMAGE_SUBCLASS'] = 'IMAGE_INDEXED'
                            dset.attrs['IMAGE_MINMAXRANGE'] = np.array(
                                [0, 255], dtype=np.uint8)
                    else:
                        grp.create_dataset(key, data=value)
            self._current_backup_indx += 1
        COMM.barrier()


    def _singlet_comparison(self, thermo):
        """
        Check which singlet states are higher
        """
        indicators = []
        for comb in combinations(thermo, 2):
            indicators.append([comb[0][get_singlet_name(
                name)] > comb[1][get_singlet_name(name)] for name in self._singlet_names])
        return indicators


    def _system_changed_phase(self, prev_comp, comp):
        """
        Check if composition changes too much from one step to another

        :param prev_comp: Composition on the previous step
        :param comp: Composition on the current step
        """
        return np.abs(prev_comp - comp) > self._max_singlet_change


    def _get_chem_pot_dict(self, chem_pot_vec):
        """
        Returns a dictionary of chem_pot based on the values in a numby array
        """
        chem_pot_dict = {}
        for i, name in enumerate(self._singlet_names):
            chem_pot_dict[name] = chem_pot_vec[i]
        return chem_pot_dict


    def _get_rhs(self, thermo, chem_pot_array, beta):
        """
        Computes the right hand side of the phase boundary equation
        """
        num_singlets = len(self._ground_states) - 1
        matrix = np.zeros((num_singlets, num_singlets))
        energy_vector = np.zeros(num_singlets)
        for i in range(num_singlets):
            for j in range(num_singlets):
                ref_singlet = thermo[0][get_singlet_name(
                    self._singlet_names[j])]
                singlet = thermo[i +
                                 1][get_singlet_name(self._singlet_names[j])]
                matrix[i, j] = ref_singlet - singlet
            ref_energy = thermo[0]["energy"] / \
                len(self._ground_states[0]["bc"].atoms)
            energy = thermo[i + 1]["energy"] / \
                len(self._ground_states[i + 1]["bc"].atoms)
            energy_vector[i] = ref_energy - energy
        inv_matrix = np.linalg.inv(matrix)
        rhs = inv_matrix.dot(energy_vector) / beta - chem_pot_array / beta
        return rhs

    def _get_singlet_array(self, thermo):
        """
        Return array of the singlet terms
        """
        singlets = []
        for entry in thermo:
            singlets.append([entry[get_singlet_name(name)]
                             for name in self._singlet_names])
        return singlets


    def _compositions_swapped(self, thermo):
        """
        Check if the compositions overlap
        """
        assert self._ref_indicators is not None

        indicators = self._singlet_comparison(thermo)

        for list1, list2 in zip(indicators, self._ref_indicators):
            comp_swapped = True
            for ind1, ind2 in zip(list1, list2):
                if ind1 == ind2:
                    comp_swapped = False
            if comp_swapped:
                return True
        return False

    def _predict(self, ode_solution, thermo):
        """
        Preduct next composition
        """
        ref_values = []
        images = []
        singlet_array = self._get_singlet_array(thermo)

        num_states = len(self._ground_states)
        num_singlets = len(self._singlet_names)
        temperature = thermo[0]["temperature"]
        for i in range(num_states):
            ref_val_in_state = []
            for j in range(num_singlets):
                history = get_singlet_evolution(ode_solution.singlets, i, j)
                ref, img = predict_composition(
                    history, ode_solution.temperatures, temperature, \
                    singlet_array[i][j])

                ref_val_in_state.append(ref)
                images.append(img)
                self._log("Temperatures used for prediction:")
                self._log("{}".format(ode_solution.temperatures))
                self._log(
                    "Compositions system {}, singlet {}:".format(
                        i, j))
                self._log("{}".format(history))
                self._log(
                    "Predicted composition for T={}K: {}".format(
                        temperature, ref))
            ref_values.append(ref_val_in_state)
        return ref_values, images

    def _one_system_changed_phase(self, thermo, ref_values):
        """
        Check if one of the systems changed phase
        """
        singlet_array = self._get_singlet_array(thermo)
        for cur_array, ref_array in zip(singlet_array, ref_values):
            for cur_val, ref_val in zip(cur_array, ref_array):
                if self._system_changed_phase(cur_val, ref_val):
                    return True
        return False

    def _prediction_match(self, thermo, ref_values, eps=0.05):
        """
        Checks if the predicted and the computed values match
        """
        singlet_array = self._get_singlet_array(thermo)
        for cur_array, ref_array in zip(singlet_array, ref_values):
            for cur_val, ref_val in zip(cur_array, ref_array):
                if abs(cur_val - ref_val) > eps:
                    return False
        return True

    def _step(self, temperature, mc_args):
        """
        Perform one ODE step
        """
        self._log("Current temperature {}K. Current chemical_potential:"
                  " {} eV/atom".format(
                    int(temperature), mc_args["chem_potential"]))

        thermo = []
        for i, sgc in enumerate(self._sgc_obj):
            self._log("Running MC for system {}".format(i))
            sgc.T = temperature
            sgc.runMC(**mc_args)
            thermo.append(sgc.get_thermodynamic())
        return thermo

    def _init_sgc(self, init_temp, symbols, mpicomm):
        """
        Initialize the SGC MC objects
        """
        self._sgc_obj = []
        for ground_state in self._ground_states:
            self._sgc_obj.append(
                SGCMonteCarlo(
                    ground_state["bc"].atoms,
                    init_temp,
                    symbols=symbols,
                    mpicomm=mpicomm))

    def separation_line_adaptive_euler(
            self,
            init_temp=100,
            min_step=1,
            stepsize=100,
            mc_args=None,
            symbols=None,
            init_mu=None,
            Tend=None):
        """
        Solve the differential equation using adaptive euler

        :param init_temp: Initial temperature
        :param min_step: Minimum step size in kelvin
        :param stepsize: Initial stepsize in kelving
        :param mc_args: Dictionary of arguments for the MC samplers.
            See :py:meth:`cemc.mcmc.SGCMonteCarlo.runMC`
        :param init_mu: Initial chemical potential
        :param Tend: Temperature at which to stop the integration
        """
        self._set_integration_direction(init_temp, Tend)
        if mc_args is None:
            mc_args = {}

        if symbols is None:
            raise ValueError("No symbols given!")

        if init_mu is None:
            # Estimate the initial chemical potential from the
            # zero kelvin limit
            chem_pot = self._get_init_chem_pot()
        else:
            # Use the user provided chemical potential is initial value
            chem_pot = np.array(init_mu)
        if COMM.Get_size() > 1:
            mpicomm = COMM
        else:
            mpicomm = None

        calcs = CECalculators(self._ground_states)
        self._init_sgc(init_temp, symbols, mpicomm)

        # Equillibriation is required here no matter what the user
        # gives as argument
        mc_args["equil"] = True

        prev_state = {}
        temperature = init_temp
        prev_state["T"] = temperature
        prev_state["rhs"] = np.zeros_like(chem_pot)
        prev_state["chem_pot"] = np.copy(chem_pot)

        delta_temp = stepsize
        is_first = True
        eps = self._max_singlet_change / 2.0
        ode_solution = PhaseBoundarySolution()

        while abs(stepsize) > min_step:
            mc_args["chem_potential"] = self._get_chem_pot_dict(chem_pot)
            beta = 1.0 / (kB * temperature)
            thermo = self._step(temperature, mc_args)
            rhs = self._get_rhs(thermo, chem_pot, beta)

            if is_first:
                self._ref_indicators = self._singlet_comparison(thermo)

            singlet_array = self._get_singlet_array(thermo)
            comp_swapped = self._compositions_swapped(thermo)
            ref_values, images = self._predict(ode_solution, thermo)
            one_system_changed_phase = self._one_system_changed_phase(thermo, ref_values)
            match = self._prediction_match(thermo, ref_values, eps=eps)
            if ode_solution.singlets:
                small_change_since_last = self._prediction_match(thermo, ode_solution.singlets[-1])
                match = match or small_change_since_last
            converged = match and not comp_swapped and not one_system_changed_phase

            if converged:
                self._log("============================================================")
                self._log("== Converged. Proceeding to the next temperature interval ==")
                self._log("============================================================")

                delta_temp = stepsize
                prev_state["T"] = temperature
                prev_state["rhs"][:] = rhs
                prev_state["chem_pot"][:] = chem_pot
                ode_solution.append(singlet_array, temperature, chem_pot)
                backupdata = {"images": images}
                backupdata.update(ode_solution.to_dict())
                self._backup(backupdata, dsetname="iter")
            else:
                # Did not converge reset and decrease the stepsize
                temperature = prev_state["T"]
                delta_temp /= 2.0
                chem_pot[:] = prev_state["chem_pot"]
                rhs[:] = prev_state["rhs"]

                # Update the target compositions to the new ones
                self._log(
                    "Did not converge. Updating target compositions. Refining stepsize. " \
                    "New stepsize: {}K".format(delta_temp))

                self._log("Resetting system")

                # Reset the system to pure phases
                calcs.reset()

            beta_prev = 1.0 / (kB * temperature)
            temperature += delta_temp
            beta_next = 1.0 / (kB * temperature)
            dbeta = beta_next - beta_prev
            chem_pot += rhs * dbeta

            # Append the last step to the array
            if abs(delta_temp) <= min_step:
                ode_solution.append(singlet_array, temperature, chem_pot)
                break
            elif self._reached_temperature_end_point(temperature, Tend):
                ode_solution.append(singlet_array, temperature, chem_pot)
                break
            is_first = False

        res = {}
        res["msg"] = "Not able to make progress with the smalles stepsize {}K".format(
            min_step)
        res.update(ode_solution.to_dict())
        return res


def fig2rgb(fig):
    """
    Convert matplotlib figure instance to a png
    """
    fig.canvas.draw()

    # Get RGB values
    width, height = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (height, width, 3)
    greyscale = 0.2989 * buf[:, :, 0] + 0.5870 * \
        buf[:, :, 1] + 0.1140 * buf[:, :, 2]
    greyscale = greyscale.astype(np.uint8)
    return greyscale

def check_gs_argument(ground_state):
    """
    Check that the ground_state arguments contain the correct fields

    :param ground_state: Ground state structure
    """
    required_fields = ["bc", "cf", "eci"]
    keys = ground_state.keys()
    for key in keys:
        if key not in required_fields:
            raise ValueError(
                "The GS argument has to contain {} keys. Given {}".format(
                    required_fields, keys))

def predict_composition(
        comp,
        temperatures,
        target_temp, target_comp):
    """
    Performs a prediction of the next composition value based on history

    :param comp: History of compositions
    :param temperatures: History of temperatures
    :param target_temp: Temperature where the composition should be predicted
    :param target_temp: Computed composition
    """
    if len(comp) == 0:
        return target_comp, None
    elif len(comp) <= 2:
        return comp[-1], None
    elif len(comp) == 3:
        k = 2
    else:
        k = 3

    temp_lin_space = np.arange(0, len(temperatures))[::-1]
    weights = np.exp(-2.0 * temp_lin_space / len(temperatures))

    # Ad hoc smoothing parameter
    # This choice leads to the deviation from the last point being
    # maximum 0.05
    smoothing = 0.05 * np.sum(weights)

    # Weight the data sich that the last point is more important than
    # the first.
    # Impact of the first point is exp(-2) relative to the impact of the
    # last point
    sign = 1
    if temperatures[1] < temperatures[0]:
        # We have to revert the arrays
        temperatures = temperatures[::-1]
        comp = comp[::-1]
        weights = weights[::-1]
        sign = -1

    spl = UnivariateSpline(temperatures, comp, k=k, w=weights, s=smoothing)
    predicted_comp = spl(target_temp)

    rgbimage = None
    rank = COMM.Get_rank()
    if rank == 0:
        # Create a plot of how the spline performs
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(
            temperatures,
            comp,
            marker='^',
            color="black",
            label="History")
        temp_lin_space = np.linspace(np.min(temperatures), target_temp + sign*40, 100)
        pred = spl(temp_lin_space)
        axis.plot(temp_lin_space, pred, "--", label="Spline")
        axis.plot([target_temp], [target_comp], 'x', label="Computed")
        axis.set_ylabel("Singlets")
        axis.set_xlabel("Temperature (K)")
        axis.legend()
        rgbimage = fig2rgb(fig)
        plt.close("all")
    rgbimage = COMM.bcast(rgbimage, root=0)
    return predicted_comp, rgbimage

def get_singlet_evolution(singlet_history, phase_indx, singlet_indx):
    """
    Return the history of one particular singlet term in one phase
    """
    history = np.zeros(len(singlet_history))
    for i, entry in enumerate(singlet_history):
        history[i] = entry[phase_indx][singlet_indx]
    return history

def get_singlet_name(orig_name):
    """
    Returns the singlet name as stored in the thermo-dictionary
    """
    return "singlet_{}".format(orig_name)
