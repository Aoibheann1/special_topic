import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



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
        solution = solve_ivp(lambda t, u: pde_system(t, u, d1, d2, dx), (t_start, t_end), u_initial, method='BDF')

    
    # central difference BCs at x=-1 and x-1, forward difference at x=0
    def pde_system(self):
        n = len(self.u) // 2
        cv = self.u[:n]
        cc = self.u[n:]

        d2cv_dx2 = np.zeros_like(cv)
        d2cc_dx2 = np.zeros_like(cc)

        # Compute second derivative of cv and cc
        d2cv_dx2[1:-1] = (cv[:-2] - 2 * cv[1:-1] + cv[2:]) / self.dx**2
        d2cc_dx2[1:-1] = (cc[:-2] - 2 * cc[1:-1] + cc[2:]) / self.dx**2

        # Boundary conditions at x = -1
        d2cv_dx2[0] = (2 * cv[1] - 2 * cv[0]) / self.dx**2

        # Boundary conditions at x = 1
        d2cc_dx2[-1] = (2 * cv[-2] - 2 * cv[-1]) / self.dx**2

        # Boundary conditions at x = 0
        d2cv_dx2[-1] = (cc[1]/self.d2 - (self.d2+1)/self.d2 * cv[-1] + cv[-2]) / self.dx**2
        d2cc_dx2[0] = (cc[1]-(1.0+self.d2) * cc[0] + self.d2 * cv[-1]) / self.dx**2

        # Compute the time derivatives
        dcv_dt = d2cv_dx2 * self.d1
        dcc_dt = d2cc_dx2

        return np.concatenate((dcv_dt, dcc_dt))

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
