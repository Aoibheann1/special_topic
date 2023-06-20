import numpy as np
import pytest
from pde_package.pde import DiffusionPDE
from pde_package.visualisation import Visualisation
from pde_package.solver import DiffusionSolver


class TestDiffusionSolver:
    @pytest.fixture
    def diffusion_solver(self):
        a = 0.5
        d1 = 0.1
        d2 = 0.2
        len_region1 = 1.0
        len_region2 = 2.0
        t_start = 0.0
        t_end = 1.0
        n_x = 10
        c_initial = np.array([1.0, 0.0, 0.0, 0.0])
        left_bc_value = 0.0
        right_bc_value = 1.0
        left_bc_type = "dirichlet"
        right_bc_type = "dirichlet"
        return DiffusionSolver(
            a,
            d1,
            d2,
            len_region1,
            len_region2,
            t_start,
            t_end,
            n_x,
            c_initial,
            left_bc_value,
            right_bc_value,
            left_bc_type,
            right_bc_type,
        )

    def test_diffusion_solver_instance(self, diffusion_solver):
        assert diffusion_solver.a == 0.5
        assert diffusion_solver.d1 == 0.1
        assert diffusion_solver.d2 == 0.2
        assert diffusion_solver.len_region1 == 1.0
        assert diffusion_solver.len_region2 == 2.0
        assert diffusion_solver.t_start == 0.0
        assert diffusion_solver.t_end == 1.0
        assert diffusion_solver.n_x == 10
        assert diffusion_solver.dx == 0.2
        assert np.array_equal(diffusion_solver.c_initial, np.array([1.0, 0.0, 0.0, 0.0]))
        assert diffusion_solver.left_bc_value == 0.0
        assert diffusion_solver.right_bc_value == 1.0
        assert diffusion_solver.left_bc_type == "dirichlet"
        assert diffusion_solver.right_bc_type == "dirichlet"

    def test_diffusion_solver_solve_pde_system(self, diffusion_solver, mocker):
        mock_solve_pde_system = mocker.patch.object(DiffusionPDE, "solve_pde_system")
        mock_solve_pde_system.return_value = (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0.0, 1.0]),
        )
        c, t = diffusion_solver.solve_pde_system()
        assert np.array_equal(c, np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert np.array_equal(t, np.array([0.0, 1.0]))
        mock_solve_pde_system.assert_called_once_with(
            diffusion_solver.c_initial,
            [diffusion_solver.left_bc_value, diffusion_solver.right_bc_value],
            [diffusion_solver.left_bc_type, diffusion_solver.right_bc_type],
        )

    def test_diffusion_solver_visualise_solution(self, diffusion_solver, mocker):
        mock_initialise_plot = mocker.patch.object(Visualisation, "initialise_plot")
        mock_show = mocker.patch.object(Visualisation, "show")
        c = np.array([[1.0, 0.0], [0.0, 1.0]])
        t = np.array([0.0, 1.0])
        diffusion_solver.visualise_solution(c, t)
        mock_initialise_plot.assert_called_once_with()
        mock_show.assert_called_once_with()

    def test_diffusion_solver_solve_and_visualise(self, diffusion_solver, mocker):
        mock_solve_pde_system = mocker.patch.object(DiffusionSolver, "solve_pde_system")
        mock_solve_pde_system.return_value = (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0.0, 1.0]),
        )
        mock_visualise_solution = mocker.patch.object(
            DiffusionSolver, "visualise_solution"
        )
        diffusion_solver.solve_and_visualise()
        mock_solve_pde_system.assert_called_once()
        mock_visualise_solution.assert_called_once_with(
            np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 1.0])
        )

