from pde_package.solver import solve
import numpy as np

# Set up the parameters
d1 = 1e-5
d2 = 1e-12
a = 1e5
x_start = -1e-1
x_end = 1e-3
t_start = 0.0
t_end = 1e-6
n_x = 1000

# Initial conditions
c1_initial = np.ones(n_x // 2)
c2_initial = np.zeros(n_x // 2)
c_initial = np.concatenate((c1_initial, c2_initial))

solve(d1, d2, x_start, x_end, t_start, t_end, n_x, c_initial)
