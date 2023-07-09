"""docstring."""
import numpy as np
from typing import Dict
from .base import BoundaryCondition


class DirichletBC(BoundaryCondition):
    """A class representing a Dirichlet boundary condition."""

    symbol = "C"

    def apply(self, d2c_dx2: np.ndarray, c: np.ndarray, dx: float, parameters:
              Dict[str, float]):
        """Apply the Dirichlet boundary condition operation.

        Parameters
        ----------
        d2c_dx2 : numpy.ndarray
            Array of concentration derivatives.
        c : numpy.ndarray
            Array of concentration values.
        dx : float
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
