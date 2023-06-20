import numpy as np
import pytest
from scipy.integrate import solve_ivp
from pde_package.pde import DiffusionPDE
import pde_package.pde

class TestDiffusionPDE:
    @pytest.fixture
    def diffusion_pde(self):
        a1 = 0.5
        a2 = 0.3
        t_start = 0.0
        t_end = 1.0
        dx = 0.1
        return DiffusionPDE(a1, a2, t_start, t_end, dx)

    def test_diffusion_pde_instance(self, diffusion_pde):
        assert diffusion_pde.a1 == 0.5
        assert diffusion_pde.a2 == 0.3
        assert diffusion_pde.t_start == 0.0
        assert diffusion_pde.t_end == 1.0
        assert diffusion_pde.dx == 0.1
        assert diffusion_pde.bc is None

    def test_diffusion_pde_solve_pde_system(self, diffusion_pde):
        c_initial = np.array([1.0, 0.0, 0.0, 0.0])
        boundary_values = [0.0, 1.0]
        boundary_types = ["dirichlet", "dirichlet"]
        c, t = diffusion_pde.solve_pde_system(c_initial, boundary_values,
                                              boundary_types)

        assert isinstance(c, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert c.shape == (4, len(t))
        assert t[0] == 0.0
        assert t[-1] == 1.0

    def test_diffusion_pde_pde_system(self, diffusion_pde):
        t = 0.5
        c = np.array([1.0, 0.0, 0.0, 0.0])
        result = diffusion_pde.pde_system(t, c)
        expected_result = np.array([-0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected_result, atol=1e-5)

    def test_diffusion_pde_pde_system_boundary_conditions(self, diffusion_pde):
        t = 0.0
        c = np.array([1.0, 0.0, 0.0, 0.0])
        diffusion_pde.bc = [None, None]
        boundary_values = [0.0, 1.0]
        boundary_types = ["dirichlet", "dirichlet"]
        boundary_applier = diffusion_pde.bc[0].__class__.__name__
        diffusion_pde.solve_pde_system(c, boundary_values, boundary_types)
        assert isinstance(diffusion_pde.bc[0], getattr(pde_package.pde, boundary_applier))
        assert isinstance(diffusion_pde.bc[1], getattr(pde_package, boundary_applier))
        assert diffusion_pde.bc[0].value == 0.0
        assert diffusion_pde.bc[1].value == 1.0

