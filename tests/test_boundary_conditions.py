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
    BoundaryCondition,
    NeumannBC,
    DirichletBC,
    BoundaryConditionApplier
)


def test_boundary_condition_str():
    """Test the __str__() method of the BoundaryCondition class."""
    bc = BoundaryCondition(1.0)
    assert str(bc) == "The value of the boundary condition is = 1.0"


@pytest.mark.parametrize("bc, dc, c, index, dx, ans_dc",
                         [(2.0, [0.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], 0, 0.5,
                           [4.0, 0.0, 0.0, 0.0, 0.0]),
                          (2.0, [0.0] * 5, [0.0, 1.0, 2.0, 3.0, 4.0], -1, 0.5,
                           [0.0, 0.0, 0.0, 0.0, -4.0])])
def test_neumann_bc_operation(bc, dc, c, index, dx, ans_dc):
    """Test the operation() method of the NeumannBC class."""
    neumann = NeumannBC(bc)
    assert neumann.operation(dc, c, index, dx) == ans_dc


@pytest.mark.parametrize("bc, dc, c, index, ans_c, ans_dc",
                         [(2.0, [1.0, 2.0, 3.0, 4.0, 5.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0], 0,
                           [2.0, 2.0, 3.0, 4.0, 5.0],
                           [0.0, 2.0, 3.0, 4.0, 5.0]),
                          (2.0, [1.0, 2.0, 3.0, 4.0, 5.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0], -1,
                           [1.0, 2.0, 3.0, 4.0, 2.0],
                           [1.0, 2.0, 3.0, 4.0, 0.0])])
def test_dirichlet_bc_operation(bc, dc, c, index, ans_c, ans_dc):
    """Test the operation() method of the DirichletBC class."""
    dirichlet = DirichletBC(bc, index)
    dirichlet.operation(dc, c)
    assert dc == ans_dc
    assert c == ans_c


def test_boundary_condition_applier():
    """Test the BoundaryConditionApplier class."""
    bca = BoundaryConditionApplier()

    with pytest.raises(ValueError):
        bca.generate_boundary_conditions([1.0], ["neumann", "dirichlet"])

    with pytest.raises(ValueError):
        bca.generate_boundary_conditions([1.0, 2.0, 3.0],
                                         ["neumann", "dirichlet"])

    with pytest.raises(ValueError):
        bca.generate_boundary_conditions([1.0, 2.0], ["neumann", "invalid"])

    conditions = bca.generate_boundary_conditions([1.0, 2.0],
                                                  ["neumann", "dirichlet"])
    assert type(conditions[0]) == NeumannBC
    assert type(conditions[1]) == DirichletBC
    assert conditions[0].value == 1.0
    assert conditions[1].value == 2.0
