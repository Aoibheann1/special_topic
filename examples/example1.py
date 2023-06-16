from diffusion.solver.diffusion_solver import solve_and_visualise_diffusion

# Set up the parameters
d1 = 1e3
d2 = 1.0
x_start = -1.0
x_end = 1.0
t_start = 0.0
t_end = 1.0
n_x = 1000

# Solve and visualize the diffusion equation
solve_and_visualise_diffusion(d1, d2, x_start, x_end, t_start, t_end, n_x)
