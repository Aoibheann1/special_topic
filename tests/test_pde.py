import pytest 
import numpy as np
from scipy.integrate import solve_ivp
from pde_package.boundary_conditions import BoundaryCondition, NeumannBC, DirichletBC, BoundaryConditionApplier
from pde_package.pde import DiffusionPDE

def test_validate_boundary_condition_value():
    # Test that a valid boundary condition value returns True
    assert BoundaryCondition.validate_boundary_condition_value(0.0) is True

    # Test that an invalid boundary condition value returns False
    assert BoundaryCondition.validate_boundary_condition_value('invalid') is False

def test_generate_boundary_condition_instances():
    # Test that the BoundaryConditionApplier class generates boundary condition instances as expected
    boundary_values = [1.0, 0.0]
    boundary_types = ['dirichlet', 'neumann']
    boundary_applier = BoundaryConditionApplier(boundary_values, boundary_types)
    boundary_conditions = boundary_applier.generate_boundary_condition_instances()
    assert isinstance(boundary_conditions[0], DirichletBC)
    assert boundary_conditions[0].value == 1.0
    assert isinstance(boundary_conditions[1], NeumannBC)
    assert boundary_conditions[1].value == 0.0

def test_solve_pde_system():
    # Test that the DiffusionPDE class solves the PDE system accurately
    a1 = 1.0
    a2 = 1.0
    t_start = 0.0
    t_end = 1.0
    dx = 0.1
    c_initial = np.array([1.0, 0.0, 0.0, 0.0])
    boundary_values = [0.0, 0.0]
    boundary_types = ['dirichlet', 'neumann']
    pde = DiffusionPDE(a1, a2, t_start, t_end, dx)
    c, t = pde.solve_pde_system(c_initial, boundary_values, boundary_types)

    # Check that the concentration values at the final time are close to the expected values
    assert np.isclose(c[0][-1], 0.3247133, rtol=1e-5)
    assert np.isclose(c[1][-1], 0.3247133, rtol=1e-5)
    assert np.isclose(c[2][-1], 0.1752867, rtol=1e-5)
    assert np.isclose(c[3][-1], 0.1752867, rtol=1e-5)