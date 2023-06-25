"""
Pytest file for testing the boundary conditions module.

This pytest file contains test cases to ensure the proper functionality of the
boundary conditions module. The module provides classes and functions for
defining and applying boundary conditions in a partial differential equation
(PDE) solver. The test cases cover various aspects of the module, including
the boundary condition classes (BoundaryCondition, NeumannBC, DirichletBC),
the BoundaryConditionApplier class, and their associated operations.

Author: Aoibheann1
"""

import pytest
from pde_package.boundary_conditions.base import BoundaryCondition
from pde_package.boundary_conditions.applier import BoundaryConditionApplier
from pde_package.boundary_conditions.dirichlet import DirichletBC
from pde_package.boundary_conditions.neumann import NeumannBC


@pytest.mark.unit
def test_boundary_condition():
    """
    Test the BoundaryCondition base class.

    This test case verifies the proper initialization of the BoundaryCondition
    base class and ensures that the value and index attributes are set
    correctly.
    """
    bc = BoundaryCondition(value=10.0, index=0)
    assert bc.value == 10.0
    assert bc.index == 0


@pytest.mark.unit
def test_boundary_condition_str():
    """
    Test the __str__() method of the BoundaryCondition class.

    This test case checks the string representation of the NeumannBC and
    DirichletBC instances and verifies that they match the expected format.
    """
    bc1 = NeumannBC(1.0, 0)
    bc2 = DirichletBC(1.0, 1)
    assert str(bc1) == "dC/dx[0] = 1.0"
    assert str(bc2) == "C[1] = 1.0"


@pytest.mark.unit
def test_boundary_condition_validation(self):
    """
    Test validation in DirichletBC initialization.

    This test case verifies that initializing a DirichletBC instance with an
    invalid value raises a ValueError as expected.
    """
    with pytest.raises(ValueError):
        DirichletBC("invalid_value", 0)


@pytest.mark.parametrize("value, dc, c, index, dx, parameters, ans_dc",
                         [(2.0, [0.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], 0, 0.5,
                           {'a': 2.0, 'c_max': 5.0, 'len_region1': 2.0,
                            'len_region2': 3.0}, 32 / 5),
                          (2.0, [0.0] * 5, [0.0, 1.0, 2.0, 3.0, 4.0], 1, 0.5,
                           {'a': 2.0, 'c_max': 4.0, 'len_region1': 3.0,
                            'len_region2': 2.0}, -7.0)])
def test_neumann_bc_operation(value, dc, c, index, dx, parameters, ans_dc):
    """
    Test the operation() method of the NeumannBC class.

    This test case applies a Neumann boundary condition to the given diffusion
    coefficient array and concentration array and checks if the resulting dc
    array matches the expected values.
    """
    neumann = NeumannBC(value, index)
    neumann.apply(dc, c, dx, parameters)
    assert dc[-index] == ans_dc


@pytest.mark.parametrize("bc, dc, c, index, dx, params, ans_c, ans_dc",
                         [(2.0, [1.0, 2.0, 3.0, 4.0, 5.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0], 0, 0.5,
                           {'a': 2.0, 'c_max': 5.0}, 0.4, 0.0),
                          (2.0, [1.0, 2.0, 3.0, 4.0, 5.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0], 1, 0.5,
                           {'a': 2.0, 'c_max': 5.0}, 0.2, 0.0)])
def test_dirichlet_bc_operation(bc, dc, c, index, dx, params, ans_c, ans_dc):
    """
    Test the operation() method of the DirichletBC class.

    This test case applies a Dirichlet boundary condition to the given
    diffusion coefficient array and concentration array and checks if the
    resulting dc and c arrays match the expected values.
    """
    dirichlet = DirichletBC(bc, index)
    dirichlet.apply(dc, c, dx, params)
    assert dc[-index] == ans_dc
    assert c[-index] == ans_c


@pytest.mark.unit
def test_boundary_condition_applier():
    """
    Test the BoundaryConditionApplier class.

    This test case verifies the behavior of the BoundaryConditionApplier class,
    including the validation of boundary condition inputs and the generation of
    boundary condition instances.
    """
    with pytest.raises(ValueError):
        BoundaryConditionApplier([1, 2, 3], ["dirichlet", "neumann"])
    with pytest.raises(ValueError):
        BoundaryConditionApplier([1, 2], ["dirichlet", "neumann", "dirichlet"])
    with pytest.raises(ValueError):
        BoundaryConditionApplier([1, 2], ["dirichlet", "invalid"])
    bc_applier = BoundaryConditionApplier([1.0, 2.0], ["dirichlet", "neumann"])
    bc_instances = bc_applier.generate_boundary_condition_instances()

    assert len(bc_instances) == 2
    assert isinstance(bc_instances[0], DirichletBC)
    assert isinstance(bc_instances[1], NeumannBC)
