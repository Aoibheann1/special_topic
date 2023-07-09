"""Module for the Method of Lines solver for transmission diffusion PDE system.

This module defines the `MethodOfLines` class, which is responsible for
solving a transmission diffusion PDE system using the Method of Lines approach.
The class inherits from the `BaseSolver` class and provides the implementation
of the `solve_pde_system` method.

Classes:
- MethodOfLines: Class for solving the transmission diffusion PDE system using
the Method of Lines approach.

"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from .base import BaseSolver
from ..boundary_conditions.applier import BoundaryConditionApplier


class MethodOfLines(BaseSolver):
    """Class for solving a transmission diffusion PDE system."""

    def solve_pde_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
        """
        Solve the PDE system.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray]: Solution concentrations, spatial points and time
            points.
        """
        boundary_applier = BoundaryConditionApplier(self.values, self.types)
        self.bc = boundary_applier.generate_boundary_condition_instances()

        solution = solve_ivp(self.ode_system, (self.T_start, self.T_end),
                             self.c_initial, method="LSODA")

        c = solution.y
        t_dim = solution.t

        n = self.parameters['n']
        c1 = c[:n, :]
        c2 = c[n:, :]
        x1 = np.linspace(-1, -self.h, n)
        x2 = np.linspace(self.h, 1, n)
        t = (t_dim * self.parameters['len_region2'] ** 2
             / self.parameters['diffusion_coefficient2'])

        return x1, x2, c1, c2, t

    def ode_system(self, t: float, c: np.ndarray) -> np.ndarray:
        """Compute the discretised system of temporal ODEs.

        Args:
            t (float): Time value.
            c (numpy.ndarray): Concentration array.

        Returns:
            numpy.ndarray: Array of computed time derivatives.
        """
        n = self.parameters['n']
        dc = np.zeros_like(c)
        d2c1_dx2 = dc[:n]
        d2c2_dx2 = dc[n:]
        c1 = c[:n]
        c2 = c[n:]
        h_squared = self.h ** 2

        # Apply exterior boundary conditions
        self.bc[0].apply(dc, c, self.h, self.parameters)
        self.bc[-1].apply(dc, c, self.h, self.parameters)

        # Apply central difference approximation to interior points
        d2c1_dx2[1:-1] = np.diff(c1, 2) / h_squared
        d2c2_dx2[1:-1] = np.diff(c2, 2) / h_squared

        inv_1_plus_a2 = 1 / (1 + self.a2)

        # Apply interface boundary conditions
        d2c1_dx2[-1] = ((-(2 + self.a2) * inv_1_plus_a2 * c1[-1]
                        + c1[-2] + c2[0] * inv_1_plus_a2)
                        / h_squared)
        d2c2_dx2[0] = (c2[1] - (1 + 2 * self.a2) * inv_1_plus_a2
                       * c2[0] + self.a2 * inv_1_plus_a2
                       * c1[-1]) / h_squared

        dc1_dt = self.a1 * d2c1_dx2
        dc2_dt = d2c2_dx2
        return np.concatenate((dc1_dt, dc2_dt))
