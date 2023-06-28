"""Docstring."""
import numpy as np
from pde_package.solver.base import BaseSolver
from pde_package.solver.method_of_lines import MethodOfLines
import pytest
import time


class MockSolver(BaseSolver):
    """A mock solver for testing purposes."""

    def solve_pde_system(self):
        """Implement a mock solver for testing purposes."""
        pass


@pytest.fixture
def solver_parameters():
    """Fixture that returns solver parameters for testing."""
    return {
        'diffusion_coefficient1': 1.0,
        'diffusion_coefficient2': 2.0,
        'len_region1': 1.0,
        'len_region2': 2.0,
        'a': 5,
        'n': 5,
        't_start': 0.0,
        't_end': 1.0,
        'c1_initial': np.ones(5),
        'c2_initial': np.ones(5),
        'c_max': 1.0,
        'left_bc_value': 0.0,
        'right_bc_value': 1.0,
        'left_bc_type': 'dirichlet',
        'right_bc_type': 'neumann'
    }


def test_base_solver_initialization(solver_parameters):
    """Test initialization of the BaseSolver class."""
    solver = MockSolver(**solver_parameters)

    for param, value in solver.parameters.items():
        assert solver_parameters[param] == value
    assert np.isclose(solver.dx, 0.2)
    assert isinstance(solver, BaseSolver)


def test_base_solver_validation(solver_parameters):
    """Test validation of the BaseSolver class."""
    invalid_parameters = {
        'diffusion_coefficient1': 0.0,
        'diffusion_coefficient2': -1.0,
        'len_region1': -2.0,
        'len_region2': 0.0,
        'a': 0.5,
        'n': 1,
        't_start': -1.0,
        't_end': -2.0,
        'c1_initial': np.zeros(5),
        'c2_initial': np.array([]),
        'c_max': -1.0
    }

    for parameter, value in invalid_parameters.items():
        solver_params = solver_parameters
        solver_params[parameter] = value
        with pytest.raises(ValueError):
            MockSolver(**solver_params)


def test_method_of_lines_solver(solver_parameters):
    """Test the MethodOfLines solver."""
    solver = MethodOfLines(**solver_parameters)

    x1, x2, c1, c2, t = solver.solve_pde_system()

    assert isinstance(x1, np.ndarray)
    assert isinstance(x2, np.ndarray)
    assert isinstance(c1, np.ndarray)
    assert isinstance(c2, np.ndarray)
    assert isinstance(t, np.ndarray)

    assert len(x1) == solver.parameters['n']
    assert len(x2) == solver.parameters['n']
    assert len(c1) == solver.parameters['n']
    assert len(c2) == solver.parameters['n']
    assert len(t) > 0


def test_steady_state_solution():
    """Test the steady state solution."""
    solver_parameters = {
        'diffusion_coefficient1': 1e-5,
        'diffusion_coefficient2': 1e-12,
        'len_region1': 1e-1,
        'len_region2': 1e-3,
        'a': 1e5,
        'n': 100,
        't_start': 0.0,
        't_end': 1e8,
        'c1_initial': np.ones(100),
        'c2_initial': np.zeros(100),
        'c_max': 1.0,
        'left_bc_value': 0.0,
        'right_bc_value': 0.0,
        'left_bc_type': 'neumann',
        'right_bc_type': 'neumann'
    }
    steady_state_c = 1.0 / (1 + solver_parameters['a']
                            * (solver_parameters['len_region2']
                               / solver_parameters['len_region1']))

    solver = MethodOfLines(**solver_parameters)

    x1, x2, c1, c2, t = solver.solve_pde_system()

    assert np.allclose(c1[:, -1], steady_state_c, atol=1e-11)
    assert np.allclose(c2[:, -1], steady_state_c, atol=1e-11)


@pytest.mark.parametrize("n, c1_initial, c2_initial", [
    (50, np.ones(50), np.zeros(50)),
    (50, np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.0, 50)),
    (50, np.random.rand(50), np.random.rand(50))
])
def test_solver_initial_conditions(n, c1_initial, c2_initial,
                                   solver_parameters):
    """Test solver behavior with different initial conditions."""
    parameters = solver_parameters
    parameters['n'] = n
    parameters['c1_initial'] = c1_initial
    parameters['c_max'] = np.max(parameters['c1_initial'])
    parameters['c2_initial'] = c2_initial
    solver = MethodOfLines(**parameters)

    x1, x2, c1, c2, t = solver.solve_pde_system()

    assert np.allclose(c1[:, 0], parameters['c1_initial']
                       / parameters['c_max'])
    assert np.allclose(c2[:, 0], parameters['c2_initial']
                       / (solver_parameters['a'] * parameters['c_max']))


def test_solver_stability(solver_parameters):
    """Test solver stability."""
    solver = MethodOfLines(**solver_parameters)

    x1, x2, c1, c2, t = solver.solve_pde_system()

    assert np.all(c1[:, -1] >= 0) and np.all(c1[:, -1] <= 1)
    assert np.all(c2[:, -1] >= 0) and np.all(c2[:, -1] <= 1)


@pytest.mark.parametrize("diffusion_coefficient1, diffusion_coefficient2, "
                         "len_region1, len_region2, a",
                         [(1e-5, 1e-12, 1e-1, 1e-3, 1e5),
                          (5e-8, 1e-10, 3e-2, 1e-4, 9e3),
                          (5e3, 1e1, 3e2, 1e4, 9),
                          (2e-8, 3e-3, 6e-2, 1e4, 4e2)])
def test_solver_edge_cases(diffusion_coefficient1, diffusion_coefficient2,
                           len_region1, len_region2, a):
    """Test solver stability with different parameters."""
    n = 50
    steady_state_c = 1.0 / (1.0 + a * (len_region2 / len_region1))

    t_start = 0.0
    t_end = 1e3 * len_region2 ** 2 / diffusion_coefficient2
    c1_initial = np.ones(n)
    c2_initial = np.zeros(n)
    c_max = 1.0
    left_bc_value = 0.0
    right_bc_value = 0.0
    left_bc_type = "neumann"
    right_bc_type = "neumann"

    solver = MethodOfLines(
        diffusion_coefficient1, diffusion_coefficient2, len_region1,
        len_region2, a, n, t_start, t_end, c1_initial, c2_initial, c_max,
        left_bc_value, right_bc_value, left_bc_type, right_bc_type
    )

    x1, x2, c1, c2, t = solver.solve_pde_system()

    steady_state_c1 = c1[:, -1]
    steady_state_c2 = c2[:, -1]

    np.testing.assert_allclose(steady_state_c1, steady_state_c,
                               atol=max(1e-4 * steady_state_c, 1e-12))
    np.testing.assert_allclose(steady_state_c2, steady_state_c,
                               atol=max(1e-4 * steady_state_c, 1e-12))


@pytest.mark.parametrize("n, threshold", [(5, 5e-1), (50, 5e1), (500, 1e2)])
def test_solver_performance(n, threshold, solver_parameters):
    """Test solver performance."""
    parameters = solver_parameters
    parameters['n'] = n
    parameters['c1_initial'] = np.ones(n)
    parameters['c2_initial'] = np.zeros(n)
    solver = MethodOfLines(**parameters)
    start_time = time.time()
    solver.solve_pde_system()
    end_time = time.time()
    execution_time = end_time - start_time

    assert execution_time < threshold, ("Execution time exceeded threshold: "
                                        f"{execution_time} s")
