import numpy as np
from pde_package.solver import DiffusionSolver

def test_solve_pde_system():
    # Test the solve_pde_system function of the DiffusionSolver class.
    a = 0.5
    d1 = 0.1
    d2 = 0.2
    len_region1 = 0.5
    len_region2 = 0.5
    t_start = 0
    t_end = 1
    n_x = 10
    c_initial = np.zeros(n_x)
    left_bc_value = 0
    right_bc_value = 1
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'
    solver = DiffusionSolver(a, d1, d2, len_region1, len_region2, t_start, t_end, n_x, c_initial, left_bc_value, right_bc_value, left_bc_type, right_bc_type)
    c, t = solver.solve_pde_system()
    assert c.shape == (n_x, t.shape[0])
    assert t[-1] == t_end

def test_visualise_solution():
    # Test the visualise_solution function of the DiffusionSolver class.
    a = 0.5
    d1 = 0.1
    d2 = 0.2
    len_region1 = 0.5
    len_region2 = 0.5
    t_start = 0
    t_end = 1
    n_x = 10
    c_initial = np.zeros(n_x)
    left_bc_value = 0
    right_bc_value = 1
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'
    solver = DiffusionSolver(a, d1, d2, len_region1, len_region2, t_start, t_end, n_x, c_initial, left_bc_value, right_bc_value, left_bc_type, right_bc_type)
    c, t = solver.solve_pde_system()
    solver.visualise_solution(c, t)
    assert True

def test_solve_and_visualise():
    # Test the solve_and_visualise function of the DiffusionSolver class.
    a = 0.5
    d1 = 0.1
    d2 = 0.2
    len_region1 = 0.5
    len_region2 = 0.5
    t_start = 0
    t_end = 1
    n_x = 10
    c_initial = np.zeros(n_x)
    left_bc_value = 0
    right_bc_value = 1
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'
    solver = DiffusionSolver(a, d1, d2, len_region1, len_region2, t_start, t_end, n_x, c_initial, left_bc_value, right_bc_value, left_bc_type, right_bc_type)
    solver.solve_and_visualise()
    assert True