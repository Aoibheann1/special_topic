"""Example 1D transmission diffusion equation problem.

This script solves a 1D transmission diffusion partial differential equation
using the Method of Lines. It provides visualizations of the solution using
the provided visualization classes.

Usage:
example1D.py [OPTIONS]

Options:
-h, --help Show this usage message.
-o, --output FILENAME Save the plot or animation to the specified file.
-t, --time-fraction FLOAT Specify the time fraction for the plot (default: 1).

Environment Variables:
None.

Files:
None.

Parameters:
diffusion_coefficient1 (float): Coefficient for diffusion in region 1.
diffusion_coefficient2 (float): Coefficient for diffusion in region 2.
alpha (float): Partition coefficient.
len_region1 (float): Length of region 1.
len_region2 (float): Length of region 2.
t_start (float): Starting time for the solution.
t_end (float): Ending time for the solution.
n (int): Number of grid points.
c1_initial (ndarray): Initial concentration values for region 1.
c2_initial (ndarray): Initial concentration values for region 2.
left_bc_value (float): Value for the left boundary condition.
right_bc_value (float): Value for the right boundary condition.
left_bc_type (str): Type of the left boundary condition (options: Neumann, Dirichlet).
right_bc_type (str): Type of the right boundary condition (options: Neumann, Dirichlet).

Note: The script assumes the presence of the 'pde_package' module and its dependencies.

"""

from transmission_diffusion_pde.solver.method_of_lines import MethodOfLines
from pde_package.visualisation.animate import Animate
from pde_package.visualisation.plot import SpecifiedTimePlot
import numpy as np

n_grid_points = 50

parameters = {
    'diffusion_coefficient1': 1e-5,
    'diffusion_coefficient2': 1e-12,
    'a': 1e5,
    'len_region1': 1e-1,
    'len_region2': 1e-3,
    't_start': 0.0,
    't_end': 1e10,
    'n': n_grid_points,
    'c1_initial': np.ones(n_grid_points),
    'c2_initial': np.zeros(n_grid_points),
    'left_bc_value': 0.0,
    'right_bc_value': 0.0,
    'left_bc_type': 'neumann',
    'right_bc_type': 'neumann'
}
c_max = np.max(parameters['c1_initial'])

solver = MethodOfLines(**parameters, c_max=c_max)

x1, x2, c1, c2, t = solver.solve_pde_system()

plot = SpecifiedTimePlot(x1, x2, c1, c2, t)
plot.show(time_fraction=0.5)
# plot.save('plot_t_0.5.png', time_fraction=0.5)

anim = Animate(x1, x2, c1, c2, t)
anim.show()
# anim.save('animation.gif')
