from diffusion.diffusion_2regions import DiffusionEquationSolver
import numpy as np

solver = DiffusionEquationSolver(size_x=2.0, num_points_x=100, time_step=0.001, final_time=1.0, D1=1.0, D2=2.0)
initial_cv = np.ones(solver.num_points_x + 1)
initial_cc = np.zeros(solver.num_points_x + 1)
solver.set_initial_conditions(initial_cv, initial_cc)
solver.solve()
solver.plot_solution()
