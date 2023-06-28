import numpy as np
from pde_package.solver.base import BaseSolver
from pde_package.solver.method_of_lines import MethodOfLines
import pytest
import time

class MockSolver(BaseSolver):
    def solve_pde_system(self):
        # Implement a mock solver for testing purposes
        pass


def test_base_solver_initialization():
    # Create sample input values
    diffusion_coefficient1 = 1.0
    diffusion_coefficient2 = 2.0
    len_region1 = 1.0
    len_region2 = 2.0
    a = 0.5
    n = 5
    t_start = 0.0
    t_end = 1.0
    c1_initial = np.ones(n)
    c2_initial = np.ones(n)
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'

    # Initialize the BaseSolver
    solver = MockSolver(
        diffusion_coefficient1,
        diffusion_coefficient2,
        len_region1,
        len_region2,
        a,
        n,
        t_start,
        t_end,
        c1_initial,
        c2_initial,
        left_bc_value,
        right_bc_value,
        left_bc_type,
        right_bc_type
    )

    # Check the initialized attributes
    assert solver.parameters['diffusion_coefficient1'] == diffusion_coefficient1
    assert solver.parameters['diffusion_coefficient2'] == diffusion_coefficient2
    assert solver.parameters['len_region1'] == len_region1
    assert solver.parameters['len_region2'] == len_region2
    assert solver.parameters['a'] == a
    assert solver.parameters['n'] == n
    assert np.isclose(solver.dx, 0.2)
    assert isinstance(solver, BaseSolver)


def test_base_solver_validation():
    # Create sample input values with invalid parameters
    diffusion_coefficient1 = 0.0
    diffusion_coefficient2 = -1.0
    len_region1 = -2.0
    len_region2 = 3.0
    a = 0.5
    n = 5
    t_start = -1.0
    t_end = -2.0
    c1_initial = np.ones(n)
    c2_initial = np.ones(n)
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'

    # Check if ValueError is raised for invalid parameters
    with pytest.raises(ValueError):
        MockSolver(
            diffusion_coefficient1,
            diffusion_coefficient2,
            len_region1,
            len_region2,
            a,
            n,
            t_start,
            t_end,
            c1_initial,
            c2_initial,
            left_bc_value,
            right_bc_value,
            left_bc_type,
            right_bc_type
        )


def test_method_of_lines_solver():
    # Create a MethodOfLines instance
    solver = MethodOfLines(
        diffusion_coefficient1=1.0,
        diffusion_coefficient2=2.0,
        len_region1=1.0,
        len_region2=2.0,
        a=0.5,
        n=5,
        t_start=0.0,
        t_end=1.0,
        c1_initial=np.ones(5),
        c2_initial=np.ones(5),
        left_bc_value=0.0,
        right_bc_value=1.0,
        left_bc_type='dirichlet',
        right_bc_type='neumann'
    )

    # Solve the PDE system
    x1, x2, c1, c2, t = solver.solve_pde_system()

    # Perform assertions on the solution
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
    # Define the parameters for the steady state solution
    diffusion_coefficient1 = 1e-5
    diffusion_coefficient2 = 1e-12
    len_region1 = 1e-1
    len_region2 = 1e-3
    a = 1e5

    # Calculate the expected steady state solution
    steady_state_c = 1.0 / (1.0 + a * (len_region2 / len_region1))

    # Set up the solver with the steady state parameters
    n = 5
    t_start = 0.0
    t_end = 1e3 * len_region2 ** 2 / diffusion_coefficient2
    c1_initial = np.ones(n)
    c2_initial = np.zeros(n)
    left_bc_value = 0.0
    right_bc_value = 0.0
    left_bc_type = "neumann"
    right_bc_type = "neumann"

    solver = MethodOfLines(
        diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2,
        a, n, t_start, t_end, c1_initial, c2_initial,
        left_bc_value, right_bc_value, left_bc_type, right_bc_type
    )

    # Solve the PDE system
    x1, x2, c1, c2, t = solver.solve_pde_system()

    # Extract the steady-state solution at the final time point
    steady_state_c1 = c1[:, -1]
    steady_state_c2 = c2[:, -1]

    # Compare the steady-state solutions with the expected values
    np.testing.assert_allclose(steady_state_c1, steady_state_c, atol=1e-11)
    np.testing.assert_allclose(steady_state_c2, steady_state_c, atol=1e-11)


# Stability Test
@pytest.mark.parametrize(
    "diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2, a",
    [
        (1e-5, 1e-12, 1e-1, 1e-3, 1e5),
        (5e-8, 1e-10, 3e-2, 1e-4, 9e3),
        (5e3, 1e1, 3e2, 1e4, 9e-3),
        (2e-8, 3e-3, 6e-2, 1e4, 4e2),
        # Add more parameter combinations for stability testing
    ]
)
def test_solver_stability(diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2, a):
    # Test stability with different parameters
    n = 50
        # Calculate the expected steady state solution
    steady_state_c = 1.0 / (1.0 + a * (len_region2 / len_region1))

    # Set up the solver with the steady state parameters
    t_start = 0.0
    t_end = 1e3 * len_region2 ** 2 / diffusion_coefficient2
    c1_initial = np.ones(n)
    c2_initial = np.zeros(n)
    left_bc_value = 0.0
    right_bc_value = 0.0
    left_bc_type = "neumann"
    right_bc_type = "neumann"
    # Set up solver and solve the PDE system
    solver = MethodOfLines(
        diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2,
        a, n, t_start, t_end, c1_initial, c2_initial,
        left_bc_value, right_bc_value, left_bc_type, right_bc_type
    )
    # Solve the PDE system
    x1, x2, c1, c2, t = solver.solve_pde_system()

    # Extract the steady-state solution at the final time point
    steady_state_c1 = c1[:, -1]
    steady_state_c2 = c2[:, -1]
    # Compare the steady-state solutions with the expected values
    np.testing.assert_allclose(steady_state_c1, steady_state_c, atol=max(1e-4 * steady_state_c, 1e-12) )
    np.testing.assert_allclose(steady_state_c2, steady_state_c, atol=max(1e-4 * steady_state_c, 1e-12))

# Performance Test
@pytest.mark.parametrize("n, threshold", [(5, 5e-1), (100, 5e1), (100, 1e2)])
def test_solver_performance(n, threshold):
    # Test solver performance with different problem sizes
        # Define the parameters for the steady state solution
    diffusion_coefficient1 = 1e-5
    diffusion_coefficient2 = 2e-12
    len_region1 = 1e-1
    len_region2 = 3e-3
    a = 1e5

    # Set up the solver with the steady state parameters
    t_start = 0.0
    t_end = 1e3 * len_region2 ** 2 / diffusion_coefficient2
    c1_initial = np.ones(n)
    c2_initial = np.zeros(n)
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = "dirichlet"
    right_bc_type = "neumann"
    # Set up solver and solve the PDE system
    solver = MethodOfLines(
        diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2,
        a, n, t_start, t_end, c1_initial, c2_initial,
        left_bc_value, right_bc_value, left_bc_type, right_bc_type
    )
    start_time = time.time()
    solver.solve_pde_system()
    end_time = time.time()
    execution_time = end_time - start_time
    # Assert the desired performance or analyze execution time
    assert execution_time < threshold, f"Execution time exceeded threshold: {execution_time} s"

# Edge Cases Test
def test_solver_edge_cases():
    # Test solver behavior with edge cases
    # Example: Empty initial concentration arrays
    n = 50
    diffusion_coefficient1 = 1
    diffusion_coefficient2 = 2
    len_region1 = 1
    len_region2 = 3
    a = 1

    # Set up the solver with the steady state parameters
    t_start = 0.0
    t_end = 1e3 * len_region2 ** 2 / diffusion_coefficient2
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = "dirichlet"
    right_bc_type = "neumann"
    c1_initial = np.array([])
    c2_initial = np.array([])

    try:
        # Set up solver and solve the PDE system
        solver = MethodOfLines(
            diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2,
            a, n, t_start, t_end, c1_initial, c2_initial,
            left_bc_value, right_bc_value, left_bc_type, right_bc_type
        )
        solver.solve_pde_system()
        # Assert the desired behavior for edge cases
        # Add your assertions here
    except ValueError as e:
        # Handle the raised exception
        print(f"ValueError occurred: {str(e)}")


@pytest.mark.parametrize("n, c1_initial, c2_initial", [
        (50, np.ones(50), np.zeros(50)),
        (50, np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.0, 50)), 
        (50, np.array([]), np.array([])),
        (50, np.zeros(50), np.random.rand(50))  
    ])
def test_solver_initial_conditions(n, c1_initial, c2_initial):
    # Test solver behavior with different initial conditions
    diffusion_coefficient1 = 1
    diffusion_coefficient2 = 2
    len_region1 = 1
    len_region2 = 3
    a = 1
    t_start = 0.0
    t_end = 1.0
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = "dirichlet"
    right_bc_type = "neumann"

    try:
        # Set up solver and solve the PDE system
        solver = MethodOfLines(
            diffusion_coefficient1, diffusion_coefficient2, len_region1, len_region2,
            a, n, t_start, t_end, c1_initial, c2_initial,
            left_bc_value, right_bc_value, left_bc_type, right_bc_type
        )
        solver.solve_pde_system()
        # Add your assertions or checks here
    except ValueError as e:
        # Handle the raised exception
        print(f"ValueError occurred: {str(e)}")


