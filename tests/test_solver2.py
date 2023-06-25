import numpy as np
from pde_package.solver.base import BaseSolver
from pde_package.solver.method_of_lines import MethodOfLines
import pytest

@pytest.mark.unit
def test_base_solver_initialization():
    # Create sample input values
    diffusion_coefficient1 = 1.0
    diffusion_coefficient2 = 2.0
    len_region1 = 1.0
    len_region2 = 2.0
    a = 0.5
    combined_n = 10
    t_start = 0.0
    t_end = 1.0
    c1_initial = np.ones(combined_n // 2)
    c2_initial = np.ones(combined_n // 2)
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'

    # Initialize the BaseSolver
    solver = BaseSolver(
        diffusion_coefficient1,
        diffusion_coefficient2,
        len_region1,
        len_region2,
        a,
        combined_n,
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
    assert solver.parameters['combined_n'] == combined_n
    assert np.isclose(solver.dx, 0.2)
    assert solver.n == 5


@pytest.mark.unit
def test_base_solver_validation():
    # Create sample input values with invalid parameters
    diffusion_coefficient1 = 0.0
    diffusion_coefficient2 = -1.0
    len_region1 = -2.0
    len_region2 = 3.0
    a = 0.5
    combined_n = 10
    t_start = -1.0
    t_end = -2.0
    c1_initial = np.ones(combined_n // 2)
    c2_initial = np.ones(combined_n // 2)
    left_bc_value = 0.0
    right_bc_value = 1.0
    left_bc_type = 'dirichlet'
    right_bc_type = 'neumann'

    # Check if ValueError is raised for invalid parameters
    with pytest.raises(ValueError):
        BaseSolver(
            diffusion_coefficient1,
            diffusion_coefficient2,
            len_region1,
            len_region2,
            a,
            combined_n,
            t_start,
            t_end,
            c1_initial,
            c2_initial,
            left_bc_value,
            right_bc_value,
            left_bc_type,
            right_bc_type
        )


@pytest.mark.unit
def test_method_of_lines_solver():
    # Create a MethodOfLines instance
    solver = MethodOfLines(
        diffusion_coefficient1=1.0,
        diffusion_coefficient2=2.0,
        len_region1=1.0,
        len_region2=2.0,
        a=0.5,
        combined_n=10,
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
    solution = solver.solve_pde_system()

    # Perform assertions on the solution
    assert solution.shape == (6, 5)  # Check shape of solution array
    assert np.allclose(solution[0], np.ones(5))  # Check initial condition
    assert np.allclose(solution[-1], np.zeros(5))  # Check final condition
