import numpy as np
import matplotlib.pyplot as plt


class HeatEquationSolver:
    def __init__(self, size_x, size_y, num_points_x, num_points_y, time_step, final_time):
        self.size_x = size_x
        self.size_y = size_y
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.time_step = time_step
        self.final_time = final_time

        self.dx = size_x / num_points_x
        self.dy = size_y / num_points_y
        self.dt = time_step

        self.num_steps = int(final_time / time_step) + 1

        self.grid = np.zeros((num_points_x + 1, num_points_y + 1))
        self.current_time = 0.0

    def set_initial_condition(self, initial_condition):
        self.grid = initial_condition

    def solve(self):
        for _ in range(self.num_steps):
            self.update()

    def update(self):
        next_grid = self.grid.copy()

        for i in range(1, self.num_points_x):
            for j in range(1, self.num_points_y):
                next_grid[i, j] = self.grid[i, j] + self.dt * self.diffusion_term(i, j)

        self.grid = next_grid
        self.current_time += self.dt

    def diffusion_term(self, i, j):
        return (self.grid[i+1, j] - 2 * self.grid[i, j] + self.grid[i-1, j]) / self.dx**2 \
               + (self.grid[i, j+1] - 2 * self.grid[i, j] + self.grid[i, j-1]) / self.dy**2

    def plot_solution(self):
        X, Y = np.meshgrid(
            np.linspace(0, self.size_x, self.num_points_x + 1),
            np.linspace(0, self.size_y, self.num_points_y + 1)
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.grid, cmap='coolwarm')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Temperature')
        plt.show()
