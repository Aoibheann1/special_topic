from scipy.integrate import solve_ivp
from .diffusion_eq import Diffusion
import numpy as np


class Simulation:
    def __init__(self, a1, a2, x_start, x_end, t_start, t_end, n_x):
        self.a1 = a1
        self.a2 = a2
        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.n_x = n_x

    def solve_pde(self):
        dx = (self.x_end - self.x_start) / self.n_x

        # Initial conditions
        C1_initial = np.ones(self.n_x // 2)
        C2_initial = np.zeros(self.n_x // 2)
        C_initial = np.concatenate((C1_initial, C2_initial))

        # Create instance of Diffusion class
        diffusion = Diffusion(self.a1, self.a2, dx)
        t_array = []

        def time_steps(t, y):
            t_array.append(t)
        # Solve the PDE system
        solution = solve_ivp(lambda t, C: diffusion.pde_system(None, t, C),
                             (self.t_start, self.t_end), C_initial, callback=time_steps,
                             method='BDF')

        # Extract the solution
        x = np.linspace(self.x_start, self.x_end, self.n_x)
        C1_solution = solution.y[:self.n_x // 2, :]
        C2_solution = solution.y[self.n_x // 2:, :]
        t = np.array(t_array)
        return x, t, C1_solution, C2_solution
