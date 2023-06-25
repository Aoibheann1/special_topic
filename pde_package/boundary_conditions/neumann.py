"""docstring."""
import numpy as np
from typing import Dict
from .base import BoundaryCondition


class NeumannBC(BoundaryCondition):
    """A class representing a Neumann boundary condition."""

    symbol = "dC/dx"

    def apply(self, dc: np.ndarray, c: np.ndarray, dx: float, parameters:
              Dict[str, float]) -> np.ndarray:
        """Apply the Neumann boundary condition.

        Parameters
        ----------
        dc : numpy.ndarray
            Array of concentration derivatives.
        c : numpy.ndarray
            Array of concentration values.
        dx : float
            Step size.
        parameters : dict
            Dictionary of boundary parameters.

        """
        regions = [parameters['len_region1'], parameters['len_region2']]
        scaled_value = (self.value * regions[self.index] / (parameters['c_max']
                                                            * parameters['a']
                                                            ** self.index))
        dc[-self.index] = (2 * c[-3 * self.index + 1] - 2 * c[-self.index]
                           - (-1) ** self.index * dx * scaled_value) / dx ** 2
