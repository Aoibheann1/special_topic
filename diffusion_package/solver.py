solver = PDESolver(x_start, x_stop, t_start, t_stop, num_points_x, num_points_t)
x, t, C = solver.solve_pde(a1, a2, bc_left_value, bc_right_value, "dirichlet")

solver = PDESolver(x_start, x_stop, t_start, t_stop, num_points_x, num_points_t)
x, t, C = solver.solve_pde(a1, a2, bc_left_value, bc_right_value, "neumann")