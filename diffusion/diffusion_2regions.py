import numpy as np
import matplotlib.pyplot as plt


class DiffusionEquationSolver:
    def __init__(self, size_x, num_points_x, time_step, final_time, D1, D2):
        self.size_x = size_x
        self.num_points_x = num_points_x
        self.time_step = time_step
        self.final_time = final_time
        self.D1 = D1
        self.D2 = D2

        self.dx = size_x / num_points_x
        self.dt = time_step

        self.num_steps = int(final_time / time_step) + 1

        self.grid_cv = np.zeros(num_points_x)
        self.grid_cc = np.zeros(num_points_x)
        self.current_time = 0.0

    def set_initial_conditions(self, initial_cv, initial_cc):
        self.grid_cv = initial_cv
        self.grid_cc = initial_cc

    def solve(self):
        for _ in range(self.num_steps):
            self.update()

    def update(self):
        next_grid_cv = self.grid_cv.copy()
        next_grid_cc = self.grid_cc.copy()

        for i in range(1, self.num_points_x - 1):
            next_grid_cv[i] = self.grid_cv[i] + self.dt * self.diffusion_term_D1(i)
            next_grid_cc[i] = self.grid_cc[i] + self.dt * self.diffusion_term_D2(i)
        next_grid_cv[-1] = self.grid_cv[-1] + self.dt * (self.grid_cc[1]/self.D2 + self.grid_cv[-2] - (self.D2+1.0)/self.D2 * self.grid_cv[-1]) # D2 * dcv/dx = dcc/dx at x = 0

        # Apply boundary conditions
        next_grid_cv[0] = next_grid_cv[1]  # dcv/dx = 0 at x = -1
        next_grid_cc[-1] = next_grid_cc[-2] # dcc/dx = 0 at x = 1
        next_grid_cc[0] = next_grid_cv[-1]  # dcv/dx = 0 at x = -1

        self.grid_cv = next_grid_cv
        self.grid_cc = next_grid_cc
        self.current_time += self.dt

    def diffusion_term_D1(self, i):
        return self.D1 * (self.grid_cv[i+1] - 2 * self.grid_cv[i] + self.grid_cv[i-1]) / self.dx**2

    def diffusion_term_D2(self, i):
        return (self.grid_cc[i+1] - 2 * self.grid_cc[i] + self.grid_cc[i-1]) / self.dx**2

    def plot_solution(self):
        x1 = np.linspace(-self.size_x/2, 0, self.num_points_x)
        x2 = np.linspace(0, self.size_x/2, self.num_points_x)

        plt.plot(x1, self.grid_cv, label='cv')
        plt.plot(x2, self.grid_cc, label='cc')
        plt.xlabel('x')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()
