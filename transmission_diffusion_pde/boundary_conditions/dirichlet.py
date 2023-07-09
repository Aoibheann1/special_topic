"""Module for a Dirichlet boundary condition.

This module defines the `DirichletBC` class, which represents a Dirichlet
boundary condition.

Classes:
- DirichletBC: A class representing a Dirichlet boundary condition.

Exceptions:
- ValueError: Raised when the Dirichlet boundary condition value is negative.

"""

import numpy as np
from typing import Dict
from .base import BoundaryCondition


class DirichletBC(BoundaryCondition):
    """A class representing a Dirichlet boundary condition."""

    symbol = "C"

    def apply(self, d2c_dx2: np.ndarray, c: np.ndarray, h: float, parameters:
              Dict[str, float]):
        """Apply the Dirichlet boundary condition operation.

        Parameters
        ----------
        d2c_dx2 : numpy.ndarray
            Array of concentration derivatives.
        c : numpy.ndarray
            Array of concentration values.
        h : float
            Step size.
        parameters : dict
            Dictionary of PDE parameters.
        """
        if self.value < 0:
            raise ValueError("Dirichlet boundary condition value cannot be "
                             "negative.")

        d2c_dx2[-self.index] = 0
        c[-self.index] = self.value / (parameters['a'] ** self.index
                                       * parameters['c_max'])
