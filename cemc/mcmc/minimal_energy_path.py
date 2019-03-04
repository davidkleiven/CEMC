from cemc.mcmc import ReactionCrdRangeConstraint
from ase.io.trajectory import Trajectory
from cemc.mcmc import CanNotFindLegalMoveError
import numpy as np


class MinimalEnergyPath(object):
    def __init__(self, mc_obj=None, observer=None, value_name="",
                 relax_steps=1000, search_steps=1000,
                 traj_file="minimal_energy_path.traj",
                 max_reac_crd=1.0):
        self.mc_obj = mc_obj
        self.observer = observer
        self.energy = []
        self.reac_crd = []
        self.value_name = value_name
        self.current_value = self.observer.get_current_value()[self.value_name]
        self.relax_steps = relax_steps
        self.constraint = ReactionCrdRangeConstraint(
            self.observer, value_name=self.value_name)
        self.mc_obj.add_constraint(self.constraint)
        self.tol = 1E-4
        self.constraint.update_range([self.current_value-self.tol,
                                      self.current_value+self.tol])
        self.search_steps = search_steps
        self.traj = Trajectory(traj_file, mode="a")
        self.max_reac_crd = max_reac_crd

    def relax(self):
        try:
            for _ in range(self.relax_steps):
                self.mc_obj._mc_step()
        except CanNotFindLegalMoveError:
            pass
        self.energy.append(self.mc_obj.current_energy)
        self.reac_crd.append(self.current_value)
        self.traj.write(self.mc_obj.atoms)

    def find_new_config(self):
        # Try for the same abount of steps to find
        # a new configuration that increases the reaction
        # coordinate
        best_trial_move = []
        min_energy_change = np.inf
        calc = self.mc_obj.atoms.get_calculator()
        current_energy = self.mc_obj.current_energy

        for i in range(self.search_steps):
            trial_move = self.mc_obj._get_trial_move()
            new_value = self.observer(trial_move, peak=True)[self.value_name]

            if new_value > self.current_value:
                energy = calc.calculate(None, ["energy"], trial_move)

                if energy - current_energy < min_energy_change:
                    best_trial_move = trial_move
                    min_energy_change = energy - current_energy
                calc.undo_changes()

        # Update the system with the newest trial move
        self.mc_obj.current_energy = calc.calculate(None, ["energy"],
                                                    best_trial_move)
        new_reac_crd = self.observer(best_trial_move)[self.value_name]
        calc.clear_history()

        if abs(new_reac_crd - self.current_value) < self.tol:
            raise RuntimeError("Could not find a configuration with a higher "
                               "value for the reaction coordiante!")
        self.current_value = new_reac_crd
        self.constraint.update_range([self.current_value - self.tol,
                                      self.current_value + self.tol])

    def log(self, msg):
        print(msg)

    def run(self):
        while self.current_value < self.max_reac_crd:
            self.log("Current reac crd: {}".format(self.current_value))
            self.relax()
            self.find_new_config()
