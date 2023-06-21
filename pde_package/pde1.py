"""PDE module."""
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple

from .boundary_conditions.applier import BoundaryConditionApplier


class DiffusionPDE:
    """Class for solving a diffusion PDE system."""

    def __init__(self, a1: float, a2: float, t_start: float, t_end: float,
                 dx: float):
        """
        Initialise the DiffusionPDE instance.

        Args:
            a1 (float): Coefficient 1.
            a2 (float): Coefficient 2.
            t_start (float): Start time.
            t_end (float): End time.
            dx (float): Step size.
        """
        self.a1 = a1
        self.a2 = a2
        self.t_start = t_start
        self.t_end = t_end
        self.dx = dx
        self.bc = None

    def solve_pde_system(self, c_initial: np.ndarray, boundary_values:
                         List[float], boundary_types: List[str]
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the PDE system.

        Args:
            c_initial (numpy.ndarray): Initial concentration array.
            boundary_values (List[float]): List of boundary values.
            boundary_types (List[str]): List of boundary condition types.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Solution concentrations and
            time points.
        """
        boundary_applier = BoundaryConditionApplier(boundary_values,
                                                    boundary_types)
        self.bc = boundary_applier.generate_boundary_condition_instances()

        # Solve the PDE system
        solution = solve_ivp(self.pde_system, (self.t_start, self.t_end),
                             c_initial, method="BDF")

        # Extract the solution
        c = solution.y
        t = solution.t
        return c, t

    def pde_system(self, t: float, c: np.ndarray) -> np.ndarray:
        """
        Compute the partial differential equation (PDE) system.

        Args:
            t (float): Time value.
            c (numpy.ndarray): Concentration array.

        Returns:
            numpy.ndarray: Array of computed time derivatives.
        """
        n = len(c) // 2
        c1 = c[:n]
        c2 = c[n:]

        dc = np.zeros_like(c)
        d2c1_dx2 = dc[:n]
        d2c2_dx2 = dc[n:]

        # Boundary conditions at x = -1
        self.bc[0].apply(dc, c, self.dx)

        # Boundary conditions at x = 1
        self.bc[-1].apply(dc, c, self.dx)

        # Compute second derivative of C1 and C2 using central difference
        d2c1_dx2[1:-1] = (c1[:-2] - 2 * c1[1:-1] + c1[2:]) / self.dx ** 2
        d2c2_dx2[1:-1] = (c2[:-2] - 2 * c2[1:-1] + c2[2:]) / self.dx ** 2

        # Boundary conditions at x = 0
        d2c1_dx2[-1] = (-(2 + self.a2) / (1 + self.a2) * c1[-1]
                        + c1[-2] + c2[0] / (1 + self.a2)) / self.dx ** 2
        d2c2_dx2[0] = (c2[1] - (1 + 2 * self.a2) / (1 + self.a2) * c2[0]
                       + self.a2 / (1 + self.a2) * c1[-1]) / self.dx ** 2

        # Compute the time derivatives
        dc1_dt = self.a1 * d2c1_dx2
        dc2_dt = d2c2_dx2

        return np.concatenate((dc1_dt, dc2_dt))
