from diffusion_package.diffusion_solver import solve

# Set up the parameters
d1 = 1e-5
d2 = 1e-12
a = 1e5
x_start = -0.1
x_end = 0.001
t_start = 0.0
t_end = 1.0
n_x = 1000

# Solve and visualize the diffusion equation
solve(d1, d2, a, x_start, x_end, t_start, t_end, n_x)
