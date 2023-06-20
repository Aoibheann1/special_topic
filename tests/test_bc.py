import numpy as np
import pytest
from pde_package.boundary_conditions import (
    BoundaryCondition,
    NeumannBC,
    DirichletBC,
    BoundaryConditionApplier,
)


class TestBoundaryCondition:
    def test_boundary_condition_instance(self):
        bc = BoundaryCondition(10.0, 0)
        assert bc.value == 10.0
        assert bc.index == 0

    def test_boundary_condition_instance_invalid_value(self):
        with pytest.raises(ValueError):
            BoundaryCondition("invalid", 0)

    def test_neumann_bc_apply(self):
        dc = np.zeros(5)
        c = np.ones(5)
        dx = 0.1
        neumann_bc = NeumannBC(1.0, 0)
        neumann_bc.apply(dc, c, dx)
        expected_result = np.array([-10.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(dc, expected_result)

    def test_dirichlet_bc_apply(self):
        dc = np.zeros(5)
        c = np.ones(5)
        dx = 0.1
        dirichlet_bc = DirichletBC(2.0, 1)
        dirichlet_bc.apply(dc, c, dx)
        expected_dc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        expected_c = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
        np.testing.assert_allclose(dc, expected_dc)
        np.testing.assert_allclose(c, expected_c)

    def test_boundary_condition_applier(self):
        values = [1.0, 2.0]
        types = ["neumann", "dirichlet"]
        applier = BoundaryConditionApplier(values, types)
        bcs = applier.generate_boundary_condition_instances()
        assert isinstance(bcs[0], NeumannBC)
        assert isinstance(bcs[1], DirichletBC)
        assert bcs[0].value == 1.0
        assert bcs[0].index == 0
        assert bcs[1].value == 2.0
        assert bcs[1].index == 1

    def test_boundary_condition_applier_invalid_values(self):
        values = [1.0]
        types = ["neumann", "dirichlet"]
        applier = BoundaryConditionApplier(values, types)
        with pytest.raises(ValueError):
            applier.generate_boundary_condition_instances()

    def test_boundary_condition_applier_invalid_types(self):
        values = [1.0, 2.0]
        types = ["invalid", "dirichlet"]
        applier = BoundaryConditionApplier(values, types)
        with pytest.raises(ValueError):
            applier.generate_boundary_condition_instances()


