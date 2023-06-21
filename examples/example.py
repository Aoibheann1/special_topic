from pde_package.solver import DiffusionSolver
from pde_package.visualisation.animate import Animate
from pde_package.visualisation.plot import SpecifiedTimePlot
import numpy as np

# Set up the parameters
parameters = {
    'd1': 1e-5,
    'd2': 1e-12,
    'a': 1e5,
    'len_region1': 1e-1,
    'len_region2': 1e-3,
    't_start': 0.0,
    't_end': 1e-6,
    'n_x': 1000,
    'left_bc_value': 0.0,
    'right_bc_value': 0.0,
    'left_bc_type': 'neumann',
    'right_bc_type': 'neumann'
}

# Initial conditions
n_points = parameters['n_x']
c1_initial = np.ones(n_points // 2)
c2_initial = np.zeros(n_points // 2)

# Create an instance of DiffusionSolver
solver = DiffusionSolver(**parameters, c1_initial=c1_initial, c2_initial = c2_initial)

# Solve the PDE and visualize the solution at a specified time
x1, x2, c1, c2, t = solver.solve_diffusion_system()
plot = SpecifiedTimePlot(x1, x2, c1, c2, t)
plot.show(time_index=3)
# plot.save('plot_t_5.png', time_index=5)

# Animate the solution
anim = Animate(x1, x2, c1, c2, t)
anim.show()
# anim.save('animation.png')
