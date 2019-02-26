import numpy as np


class GradientCoefficientRhsBuilder(object):
    def __init__(self, boundary_values):
        self.boundary_values = boundary_values

    def grad(self, x):
        raise NotImplementedError("Sub-classes has to implement this method!")

    def evaluate(self, x):
        raise NotImplementedError("Has to be implemented in subclasses!")

    def _get_varying_and_fixed_params(self, interface):
        varying_params = self._varying_parameters(interface)
        fixed = {}
        for i, v in enumerate(self.boundary_values[interface]):
            if i in varying_params:
                continue
            fixed[i] = v[0]
        return varying_params, fixed

    def construct_rhs_and_boundary(self, interface):
        """Construct the right hand side of the Euler equations
            for a given interface."""
        varying_params, fixed = self._get_varying_and_fixed_params(interface)
        b_vals = [self.boundary_values[interface][x] for x in varying_params]
        return ProjectedGradient(varying_params, fixed, self), b_vals

    def get_projected(self, interface):
        varying_params, fixed = self._get_varying_and_fixed_params(interface)
        return ProjectedFunc(varying_params, fixed, self)

    def _varying_parameters(self, interface):
        varying = []
        for i, v in enumerate(self.boundary_values[interface]):
            if abs(v[0] - v[1]) > 1E-6:
                varying.append(i)
        return varying


class ProjectedFunction(object):
    def __init__(self, varying_vars, fixed_vars, rhs_builder):
        self.fixed_vars = fixed_vars
        self.varying_vars = varying_vars
        self.rhs_builder = rhs_builder

    @property
    def tot_num_vars(self):
        return len(self.varying_vars) + len(self.fixed_vars)

    def get_full_x(self, x):
        """Evaluate the projected function."""
        X = np.zeros((self.tot_num_vars, x.shape[1]))
        for i, var in enumerate(self.varying_vars):
            X[var, :] = x[2*i+1, :]

        for k, v in self.fixed_vars:
            X[k, :] = v

        return X

    def __call__(self, x):
        raise NotImplementedError("Has to be implemented in derived classes!")


class ProjectedGradient(ProjectedFunction):
    def __init__(self, varying_vars, fixed_vars, rhs_builder):
        ProjectedFunction.__init__(self, varying_vars, fixed_vars, rhs_builder)
        self.active_index = 0

    def __getitem__(self, index):
        self.active_index = index
        if index >= len(self.varying_vars):
            raise IndexError("Cannot calculate gradient component exceeded "
                             "the number of varying parameters!")
        return self

    def __call__(self, x):
        X = self.get_full_x(x)
        full_solution = self.rhs_builder.grad(X)
        return full_solution[self.varying_vars[self.active_index], :]


class ProjectedFunc(ProjectedFunction):
    def __init__(self, varying_vars, fixed_vars, rhs_builder):
        ProjectedFunction.__init__(self, varying_vars, fixed_vars, rhs_builder)

    def __call__(self, x):
        X = self.get_full_x(x)
        return self.rhs_builder.evaluate(X)
