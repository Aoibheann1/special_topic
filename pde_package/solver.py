"""Solver module."""
from .pde.diffusion import DiffusionPDE
import numpy as np


class DiffusionSolver:
    """Class for solving and visualising the diffusion equation."""

    def __init__(
        self,
        a: float,
        d1: float,
        d2: float,
        len_region1: float,
        len_region2: float,
        t_start: float,
        t_end: float,
        n_x: int,
        c_initial: np.ndarray,
        left_bc_value: float,
        right_bc_value: float,
        left_bc_type: str,
        right_bc_type: str,
    ):
        """
        Initialise the DiffusionSolver instance.

        Args:
            a (float): Parameter a.
            d1 (float): Parameter d1.
            d2 (float): Parameter d2.
            len_region1 (float): Length of region 1.
            len_region2 (float): Length of region 2.
            t_start (float): Start time value.
            t_end (float): End time value.
            n_x (int): Number of x points.
            c_initial (numpy.ndarray): Initial concentration values.
            left_bc_value (float): Value of the left boundary condition.
            right_bc_value (float): Value of the right boundary condition.
            left_bc_type (str): Type of the left boundary condition.
            right_bc_type (str): Type of the right boundary condition.
        """
        self.a = a
        self.d1 = d1
        self.d2 = d2
        self.len_region1 = len_region1
        self.len_region2 = len_region2
        self.t_start = t_start
        self.t_end = t_end
        self.n_x = n_x
        self.dx = 2.0 / self.n_x
        self.c_initial = c_initial
        self.left_bc_value = left_bc_value
        self.right_bc_value = right_bc_value
        self.left_bc_type = left_bc_type
        self.right_bc_type = right_bc_type

    def solve_diffusion_system(self):
        """
        Solve the diffusion equation system.

        Returns:
            numpy.ndarray: Array of concentration values.
            numpy.ndarray: Array of time values.
        """
        a1 = self.d1 / self.d2 * (self.len_region2 / self.len_region1) ** 2
        a2 = self.d1 / self.d2 * (self.len_region2 / self.len_region1) / self.a
        values = [self.left_bc_value, self.right_bc_value]
        types = [self.left_bc_type, self.right_bc_type]
        pde = DiffusionPDE(a1, a2, self.t_start, self.t_end, self.dx)
        c, t = pde.solve_pde_system(self.c_initial, values, types)

        c1 = c[:self.n_x // 2, :]
        c2 = c[self.n_x // 2:, :]
        x1 = np.linspace(-1, -self.dx, self.n_x // 2)
        x2 = np.linspace(self.dx, 1, self.n_x // 2)
        return x1, x2, c1, c2, t
