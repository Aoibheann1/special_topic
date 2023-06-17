from functools import singledispatch
import numpy as np


class Diffusion:
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def pde_system(self, t, u, dx):
        raise NotImplementedError("Subclasses must implement pde_system method")

    def solve(self, t_start, t_end, x_start, x_end, n_x):
        dx = (x_end - x_start) / n_x

        # Initialize the solution array
        cv_initial = np.ones(n_x // 2)
        cc_initial = np.zeros(n_x // 2)
        u_initial = np.concatenate((cv_initial, cc_initial))
        self.u_initial = u_initial

        # Initialize the derivatives array
        self.d2cv_dx2 = np.zeros_like(cv_initial)
        self.d2cc_dx2 = np.zeros_like(cc_initial)

        # Solve the PDE system
        solution = solve_ivp(lambda t, u: self.pde_system(t, u, dx), (t_start, t_end), u_initial, method='BDF')

        # Extract the solution
        x = np.linspace(x_start, x_end, n_x)
        cv_solution = solution.y[:n_x // 2, :]
        cc_solution = solution.y[n_x // 2:, :]
        m = len(cv_solution[0, :])
        t = np.linspace(t_start, t_end, m)

        return x, cv_solution, cc_solution, t


@singledispatch
def apply_boundary_conditions(diffusion_solver, cv_bc_left, cv_bc_right, cc_bc_left, cc_bc_right):
    raise NotImplementedError("Boundary condition type not supported")


@apply_boundary_conditions.register
def apply_neumann_boundary_conditions(diffusion_solver, cv_bc_left, cv_bc_right, cc_bc_left, cc_bc_right):
    n = len(diffusion_solver.u_initial) // 2
    diffusion_solver.d2cv_dx2[0] = (cv_bc_left - diffusion_solver.u_initial[0]) / diffusion_solver.dx
    diffusion_solver.d2cc_dx2[-1] = (cc_bc_right - diffusion_solver.u_initial[-1]) / diffusion_solver.dx


class MyDiffusion(Diffusion):
    def __init__(self, d1, d2, cv_bc_left, cv_bc_right, cc_bc_left, cc_bc_right):
        super().__init__(d1, d2)
        self.cv_bc_left = cv_bc_left
        self.cv_bc_right = cv_bc_right
        self.cc_bc_left = cc_bc_left
        self.cc_bc_right = cc_bc_right

    def pde_system(self, t, u, dx):
        n = len(u) // 2
        cv = u[:n]
        cc = u[n:]

        self.d2cv_dx2[1:-1] = (cv[:-2] - 2 * cv[1:-1] + cv[2:]) / dx**2
        self.d2cc_dx2[1:-1] = (cc[:-2] - 2 * cc[1:-1] + cc[2:]) / dx**2

        apply_boundary_conditions(self, self.cv_bc_left, self.cv_bc_right, self.cc_bc_left, self.cc_bc_right)

        dcv_dt = self.d2cv_dx2 * self.d1
        dcc_dt = self.d2cc_dx2

        return np.concatenate((dcv_dt, dcc_dt))
