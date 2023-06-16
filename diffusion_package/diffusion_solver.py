from .simulation import Simulation
from .visualisation import Visualisation


def solve(D1, D2, a, x_start, x_end, t_start, t_end, n_x):
    # Solve the diffusion equation
    simulation = Simulation(D1, D2, a, x_start, x_end, t_start, t_end, n_x)
    x, t, C1_solution, C2_solution = simulation.solve_pde()

    # Visualise the solution
    visualisation = Visualisation(x, C1_solution, C2_solution, t)
    visualisation.animate()
    visualisation.show()
