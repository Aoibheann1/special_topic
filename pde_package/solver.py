from .pde import PDE
from .visualisation import Visualisation
import numpy as np

def solve(a, d1, d2, x_start, x_end, t_start, t_end, n_x, c_initial):
    a1 = d1 / d2 * (x_end / x_start) ** 2
    a2 = d1 / d2 * abs(x_end / x_start) / a
    dx = 2.0 / n_x

    # Solve the diffusion equation
    pde = PDE(a1, a2, t_start, t_end, dx)
    c, t = pde.solve_pde(c_initial)
    c1 = c[:n_x // 2, :]
    c2 = c[n_x // 2:, :]
    x1 = np.linspace(x_start, dx, n_x // 2)
    x2 = np.linspace(dx, x_end, n_x // 2)


    # Visualise the solution
    visualisation = Visualisation(x1, x2, c1, c2, t)
    visualisation.animate()
    visualisation.show()