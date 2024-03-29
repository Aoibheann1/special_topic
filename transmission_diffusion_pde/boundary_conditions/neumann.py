"""Module for a Neumann boundary condition.

This module defines the `NeumannBC` class, which represents a Neumann boundary
condition.

Classes:
- NeumannBC: A class representing a Neumann boundary condition.

Exceptions:
- None.

"""

import numpy as np
from typing import Dict
from .base import BoundaryCondition


class NeumannBC(BoundaryCondition):
    """A class representing a Neumann boundary condition."""

    symbol = "dC/dx"

    def apply(self, d2c_dx2: np.ndarray, c: np.ndarray, h: float, parameters:
              Dict[str, float]):
        """Apply the Neumann boundary condition.

        Parameters
        ----------
        d2c_dx2 : numpy.ndarray
            Array of concentration derivatives.
        c : numpy.ndarray
            Array of concentration values.
        h : float
            Step size.
        parameters : dict
            Dictionary of boundary parameters.

        """
        regions = [parameters['len_region1'], parameters['len_region2']]
        scaled_value = (
            self.value * regions[self.index]
            / (parameters['c_max'] * parameters['a'] ** self.index)
            )
        d2c_dx2[-self.index] = (
            (2 * c[-3 * self.index + 1] - 2 * c[-self.index]
             - (-1) ** self.index * h * scaled_value) / h ** 2
        )
