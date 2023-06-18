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
        self.dx = (self.x_end - self.x_start) / self.n_x

    def check_bcs(self, bc_left_value, bc_right_value, bc_left_type, bc_right_type):
        if bc_left_type == "dirichlet":
            bc_left = DirichletBoundaryCondition(bc_left_value)
        elif bc_left_type == "neumann":
            bc_left = NeumannBoundaryCondition(bc_left_value, self.dx)
        else:
            raise ValueError("Invalid left boundary condition type specified")  
        if bc_right_type == "dirichlet":
            bc_right = DirichletBoundaryCondition(bc_right_value, self.dx)
        elif bc_right_type == "neumann":
            bc_right = NeumannBoundaryCondition(bc_right_value, self.dx)
        else:
            raise ValueError("Invalid right boundary condition type specified")
        return bc_left, bc_right

    def solve_pde(self, bc_left_value, bc_right_value, bc_left_type, bc_right_type):
        dx = (self.x_end - self.x_start) / self.n_x

        # Initial conditions
        C1_initial = np.ones(self.n_x // 2)
        C2_initial = np.zeros(self.n_x // 2)
        C_initial = np.concatenate((C1_initial, C2_initial))

        bc_left, bc_right = self.check_bcs(self, bc_left_value, bc_right_value, bc_left_type, bc_right_type)


        # Create instance of Diffusion class
        diffusion = Diffusion(self.a1, self.a2, dx)
        pde_system = diffusion.discretize(bc_left, bc_right)(self._pde_system)

        # Solve the PDE system
        pde_system = diffusion.discretize(bc_left, bc_right)(None, None, None)
        solution = solve_ivp(pde_system, (self.t_start, self.t_end), C_initial, method='BDF')

        # Extract the solution
        x = np.linspace(self.x_start, self.x_end, self.n_x)
        C1_solution = solution.y[:self.n_x // 2, :]
        C2_solution = solution.y[self.n_x // 2:, :]
        t = solution.t

        return x, t, C1_solution, C2_solution
