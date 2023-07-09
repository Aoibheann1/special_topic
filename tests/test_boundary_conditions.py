"""
Pytest file for testing the boundary_conditions subpackage.

This pytest file contains test cases to ensure the proper functionality of the
boundary_conditions subpackage. The subpackage provides classes and functions
for defining and applying boundary conditions in a transmission diffusion
PDE solver. The test cases cover various aspects of the subpackage, including
the boundary condition classes (DirichletBC, NeumannBC), the
BoundaryConditionApplier class, and their associated operations.
"""

import pytest
from transmission_diffusion_pde import (
    DirichletBC,
    NeumannBC,
    BoundaryConditionApplier,
)


def test_boundary_condition():
    """
    Test the DirichletBC and NeumannBC classes.

    This test case verifies the proper initialisation of the DirichletBC and
    NeumannBC classes and ensures that the value and index attributes are set
    correctly.
    """
    bc = DirichletBC(value=10.0, index=0)
    assert bc.value == 10.0
    assert bc.index == 0

    bc = NeumannBC(value=1.0, index=1)
    assert bc.value == 1.0
    assert bc.index == 1


def test_boundary_condition_applier():
    """
    Test the BoundaryConditionApplier class.

    This test case verifies the behavior of the BoundaryConditionApplier class,
    including the validation of boundary condition inputs and the generation of
    boundary condition instances.
    """
    values = [1.0, 2.0]
    types = ["Dirichlet", "Neumann"]
    bc_applier = BoundaryConditionApplier(values, types)
    bc_instances = bc_applier.generate_boundary_condition_instances()

    assert len(bc_instances) == 2
    assert isinstance(bc_instances[0], DirichletBC)
    assert isinstance(bc_instances[1], NeumannBC)

    with pytest.raises(ValueError):
        values = [1.0, 2.0, 3.0]
        types = ["Dirichlet", "Neumann"]
        BoundaryConditionApplier(values, types)

    with pytest.raises(ValueError):
        values = [1.0, 2.0]
        types = ["Dirichlet", "Neumann", "Dirichlet"]
        BoundaryConditionApplier(values, types)

    with pytest.raises(ValueError):
        values = [1.0, 2.0]
        types = ["Dirichlet", "invalid"]
        BoundaryConditionApplier(values, types)
