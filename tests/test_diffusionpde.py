import numpy as np
import pytest
from pde_package.pde import DiffusionPDE


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
        assert np.isclose(c[0][-1], 0.0, rtol=1e-5)
        assert np.isclose(c[1][-1], 0.384615, rtol=1e-5)
        assert np.isclose(c[2][-1], 0.884615, rtol=1e-5)
        assert np.isclose(c[3][-1], 1.0, rtol=1e-5)
