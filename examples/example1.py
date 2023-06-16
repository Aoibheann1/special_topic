from diffusion_package.diffusion_solver import solve

# Set up the parameters
d1 = 1e-5
d2 = 1e-12
a = 1e5
x_start = -1e-1
x_end = 1e-3
t_start = 0.0
t_end = 1e6
n_x = 1000

a1 = d1 / d2 * (x_end / x_start) ** 2
a2 = d1 / d2 * abs(x_end / x_start) / a
# Solve and visualize the diffusion equation
solve(a1, a2, x_start, x_end, t_start, t_end, n_x)
