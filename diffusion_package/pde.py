import numpy as np
from scipy.integrate import solve_ivp
from .diffusion_eq import Diffusion
from .boundary import DirichletBoundaryCondition, NeumannBoundaryCondition

class PDESolver:
    def __init__(self, a1, a2, dx, x_start, x_end, n_x, t_start, t_end):
        self.a1 = a1
        self.a2 = a2
        self.dx = dx
        self.x_start = x_start
        self.x_end = x_end
        self.n_x = n_x
        self.t_start = t_start
        self.t_end = t_end

    def solve_pde(self, a1, a2, bc_left_value, bc_right_value, bc_type):
        dx = (self.x_end - self.x_start) / self.n_x

        # Initial conditions
        C1_initial = np.ones(self.n_x // 2)
        C2_initial = np.zeros(self.n_x // 2)
        C_initial = np.concatenate((C1_initial, C2_initial))

        if bc_type == "dirichlet":
            bc_left = DirichletBoundaryCondition(bc_left_value)
            bc_right = DirichletBoundaryCondition(bc_right_value)
        elif bc_type == "neumann":
            bc_left = NeumannBoundaryCondition(bc_left_value, dx)
            bc_right = NeumannBoundaryCondition(bc_right_value, dx)
        else:
            raise ValueError("Invalid boundary condition type specified")

        diffusion = Diffusion(a1, a2, dx)
        pde_system = diffusion.discretize(bc_left, bc_right)(self._pde_system)

        # Create instance of Diffusion class
        diffusion = Diffusion(self.a1, self.a2, dx)

        bc_left = NeumannBoundaryCondition(0)
        bc_right = DirichletBoundaryCondition(1)
        # Solve the PDE system
        pde_system = diffusion.discretize(bc_left, bc_right)(None, None, None)
        solution = solve_ivp(pde_system, (self.t_start, self.t_end), C_initial, method='BDF')

        # Extract the solution
        x = np.linspace(self.x_start, self.x_end, self.n_x)
        C1_solution = solution.y[:self.n_x // 2, :]
        C2_solution = solution.y[self.n_x // 2:, :]
        t = solution.t

        return x, t, C1_solution, C2_solution