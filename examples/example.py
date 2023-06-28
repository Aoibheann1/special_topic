from pde_package.solver.method_of_lines import MethodOfLines
from pde_package.visualisation.animate import Animate
from pde_package.visualisation.plot import SpecifiedTimePlot
import numpy as np

# Initial conditions
n_points = 50
c1_initial = np.ones(n_points)
c2_initial = np.zeros(n_points)


# Set up the parameters
parameters = {
    'diffusion_coefficient1': 1e-5,
    'diffusion_coefficient2': 1e-12,
    'a': 1e5,
    'len_region1': 1e-1,
    'len_region2': 1e-3,
    't_start': 0.0,
    't_end': 1e10,
    'n': n_points,
    'left_bc_value': 0.0,
    'right_bc_value': 0.0,
    'left_bc_type': 'neumann',
    'right_bc_type': 'neumann'
}

# Create an instance of DiffusionSolver
solver = MethodOfLines(**parameters, c1_initial = c1_initial, c2_initial = c2_initial)

# Solve the PDE and visualize the solution at a specified time
x1, x2, c1, c2, t = solver.solve_pde_system()
plot = SpecifiedTimePlot(x1, x2, c1, c2, t)
plot.show(time_fraction=0.5)
# plot.save('plot_t_5.png', time_index=5)

# Animate the solution
anim = Animate(x1, x2, c1, c2, t)
anim.show()
# anim.save('animation.png')
