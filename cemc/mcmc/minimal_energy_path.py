from cemc.mcmc import ReactionCrdRangeConstraint


class MinimalEnergyPath(object):
    def __init__(self, mc_obj=None, observer=None, reac_crd=[0.0, 1.0],
                 num_bins=100, value_name="", relax_steps=1000,
                 search_steps=1000, traj_file="minimal_energy_path.traj"):
        self.mc_obj = mc_obj
        self.observer = observer
        self.reac_crd = reac_crd
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

    def relax(self):
        for _ in range(self.relax_steps):
            self.mc_obj._mc_steps()
        self.energy.append(self.mc_obj.current_energy)
        self.reac_crd.append(self.current_value)
        self.traj.write(self.mc_obj.atoms)

    def find_new_config(self):
        # Try for the same abount of steps to find
        # a new configuration that increases the reaction
        # coordinate
        orig_temp = self.mc_obj.temp
        best_trial_move = None
        min_energy_change = np.inf
        calc = self.mc_obj.atoms.get_calculator()
        for i in range(self.search_steps):
            trial_move = self.mc_obj._get_trial_move()
            new_value = self.observer(trial_move, peak=True)[self.value_name]

            if new_value > self.current_value:
                energy = calc.calculate(None, ["energy"], trial_move)

                if energy - self.current_value < min_energy_change:
                    best_trial_move = trial_move
                    min_energy_change = energy
                calc.undo_changes()

        # Update the system with the newest trial move
        self.mc_obj.current_energy = calc.calculate(None, ["energy"],
                                                    best_trial_move)
        new_reac_crd = self.observer(best_trial_move)[self.value_name]
        self.current_value = new_reac_crd
        self.constraint.update_range([self.current_value - self.tol,
                                      self.current_value + self.tol])
