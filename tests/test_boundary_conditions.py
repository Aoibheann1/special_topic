"""
Pytest file for testing boundary conditions module.

This pytest file contains test cases to ensure the proper functionality
of the boundary conditions module. The module provides classes and functions
for defining and applying boundary conditions in a partial differential
equation (PDE) solver. The test cases cover various aspects of the module,
including the boundary condition classes (BoundaryCondition, NeumannBC,
DirichletBC), the BoundaryConditionApplier class, and their associated
operations.

Author: Aoibheann1
"""

import pytest
from pde_package.boundary_conditions import (
    NeumannBC,
    DirichletBC,
    BoundaryConditionApplier
)


def test_boundary_condition_str():
    """Test the __str__() method of the BoundaryCondition class."""
    bc1 = NeumannBC(1.0, 0)
    bc2 = DirichletBC(1.0, 1)
    assert str(bc1) == "dC/dx[0] = 1.0"
    assert str(bc2) == "C[1] = 1.0"


@pytest.mark.parametrize("value, dc, c, index, dx, ans_dc",
                         [(2.0, [0.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], 0, 0.5,
                           [4.0, 0.0, 0.0, 0.0, 0.0]),
                          (2.0, [0.0] * 5, [0.0, 1.0, 2.0, 3.0, 4.0], 1, 0.5,
                           [0.0, 0.0, 0.0, 0.0, -4.0])])
def test_neumann_bc_operation(value, dc, c, index, dx, ans_dc):
    """Test the operation() method of the NeumannBC class."""
    neumann = NeumannBC(value, index)
    assert neumann.apply(dc, c, dx) == ans_dc


@pytest.mark.parametrize("bc, dc, c, index, dx, ans_c, ans_dc",
                         [(2.0, [1.0, 2.0, 3.0, 4.0, 5.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0], 0, 0.5,
                           [2.0, 2.0, 3.0, 4.0, 5.0],
                           [0.0, 2.0, 3.0, 4.0, 5.0]),
                          (2.0, [1.0, 2.0, 3.0, 4.0, 5.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0], 1, 0.5,
                           [1.0, 2.0, 3.0, 4.0, 2.0],
                           [1.0, 2.0, 3.0, 4.0, 0.0])])
def test_dirichlet_bc_operation(bc, dc, c, index, dx, ans_c, ans_dc):
    """Test the operation() method of the DirichletBC class."""
    dirichlet = DirichletBC(bc, index)
    dirichlet.apply(dc, c, dx)
    assert dc == ans_dc
    assert c == ans_c


def test_boundary_condition_applier():
    """Test the BoundaryConditionApplier class."""
    with pytest.raises(ValueError):
        BoundaryConditionApplier([1, 2, 3], ["dirichlet", "neumann"])
    with pytest.raises(ValueError):
        BoundaryConditionApplier([1, 2], ["dirichlet", "neumann", "dirichlet"])
    with pytest.raises(ValueError):
        BoundaryConditionApplier([1, 2], ["dirichlet", "invalid"])
    bc_applier = BoundaryConditionApplier([1.0, 2.0], ["dirichlet", "neumann"])
    bc = bc_applier.generate_boundary_condition_instances()
    assert isinstance(bc[0], DirichletBC)
    assert isinstance(bc[1], NeumannBC)
