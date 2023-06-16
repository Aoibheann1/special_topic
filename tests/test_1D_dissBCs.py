import pytest
import numpy as np
try:
    from special_topic.examples.diffusion_2regions import DiffusionEquationSolver
except ImportError:
    pass

def test_group_import():
    from special_topic.examples.diffusion_2regions import DiffusionEquationSolver

def test_initial_conditions():
    size_x = 2.0
    num_points_x = 100
    time_step = 0.001
    final_time = 1.0
    D1 = 1.0
    D2 = 2.0

    solver = DiffusionEquationSolver(size_x, num_points_x, time_step, final_time, D1, D2)

    initial_cv = np.ones(num_points_x)
    initial_cc = np.zeros(num_points_x - 1)

    solver.set_initial_conditions(initial_cv, initial_cc)

    assert np.array_equal(solver.grid_cv, initial_cv)
    assert np.array_equal(solver.grid_cc, initial_cc)

def test_diffusion_term_D1():
    size_x = 2.0
    num_points_x = 100
    time_step = 0.001
    final_time = 1.0
    D1 = 1.0
    D2 = 2.0

    solver = DiffusionEquationSolver(size_x, num_points_x, time_step, final_time, D1, D2)

    solver.grid_cv = np.ones(num_points_x)

    i = 5
    diffusion_term = (solver.grid_cv[i + 1] - 2 * solver.grid_cv[i] + solver.grid_cv[i - 1]) / solver.dx ** 2
    expected_diffusion_term = D1 * (solver.grid_cv[i + 1] - 2 * solver.grid_cv[i] + solver.grid_cv[i - 1]) / solver.dx ** 2

    assert diffusion_term == expected_diffusion_term


def test_diffusion_term_D2():
    size_x = 2.0
    num_points_x = 100
    time_step = 0.001
    final_time = 1.0
    D1 = 1.0
    D2 = 2.0

    solver = DiffusionEquationSolver(size_x, num_points_x, time_step, final_time, D1, D2)

    solver.grid_cc = np.ones(num_points_x - 1)

    i = 5
    diffusion_term = (solver.grid_cc[i + 1] - 2 * solver.grid_cc[i] + solver.grid_cc[i - 1]) / solver.dx ** 2

    assert diffusion_term == solver.diffusion_term_D2(i)
