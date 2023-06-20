from scipy.integrate import solve_ivp
from .diffusion import Diffusion
import numpy as np
from boundary_conditions import BoundaryConditionApplier

class PDE:
    def __init__(self, a1, a2, t_start, t_end, dx):
        self.a1 = a1
        self.a2 = a2
        self.t_start = t_start
        self.t_end = t_end
        self.dx = dx
    def solve_pde(self, c_initial, values, types):

        bca = BoundaryConditionApplier()
        bc = bca.generate_boundary_conditions(values, types)

        # Create instance of Diffusion class
        diffusion = Diffusion(self.a1, self.a2, self.dx, bc)

        # Solve the PDE system
        solution = solve_ivp(lambda t, c: diffusion.pde_system(None, t, c),
                             (self.t_start, self.t_end), c_initial,
                             method='BDF')

        # Extract the solution
        c = solution.y
        t = solution.t
        return c, t
    